import nltk
import re
import language_tool_python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import util

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ScoringEngine:
    def __init__(self, transcript, duration, nlp_model, semantic_model, grammar_tool, sentiment_analyzer):
        self.text = transcript
        self.duration = duration
        self.nlp = nlp_model
        self.semantic_model = semantic_model
        self.grammar_tool = grammar_tool
        self.sentiment_analyzer = sentiment_analyzer

        # Pre-processing
        self.doc = self.nlp(transcript)
        self.sentences = [sent.text.strip() for sent in self.doc.sents]
        # Word Count Fix: Exclude punctuation AND spaces
        self.words = [token.text for token in self.doc if not token.is_punct and not token.is_space]
        self.word_count = len(self.words)

    def evaluate(self):
        """
        Generates a list of detailed rows for the UI table and calculates total score.
        """
        table_rows = []

        # 1. Content & Structure (Total 40)
        c_rows = self._analyze_content()
        table_rows.extend(c_rows)

        # 2. Speech Rate (Total 10)
        r_rows = self._analyze_speech_rate()
        table_rows.extend(r_rows)

        # 3. Language & Grammar (Total 20)
        g_rows = self._analyze_grammar_vocab()
        table_rows.extend(g_rows)

        # 4. Clarity (Total 15)
        cl_rows = self._analyze_clarity()
        table_rows.extend(cl_rows)

        # 5. Engagement (Total 15)
        e_rows = self._analyze_engagement()
        table_rows.extend(e_rows)

        # Calculate Overall Score (Simple Sum)
        total_score = sum(row['score'] for row in table_rows)
        
        return {
            "overall_score": round(total_score, 2),
            "table_data": table_rows,
            "meta": {
                "word_count": self.word_count,
                "duration": self.duration,
                "wpm": round((self.word_count / self.duration) * 60) if self.duration else 0
            }
        }

    def _analyze_content(self):
        rows = []
        text_lower = self.text.lower()

        # A. Salutation (Max 5)
        sal_score = 0
        sal_feedback = "Missing"
        if any(x in text_lower for x in ["excited to introduce", "feeling great"]):
            sal_score = 5; sal_feedback = "Excellent enthusiasm"
        elif any(x in text_lower for x in ["good morning", "good afternoon", "hello everyone"]):
            sal_score = 4; sal_feedback = "Good standard greeting"
        elif any(x in text_lower for x in ["hi", "hello"]):
            sal_score = 2; sal_feedback = "Basic greeting"
        
        rows.append({
            "category": "Content & Structure",
            "criteria": "Salutation",
            "score": sal_score,
            "max": 5,
            "feedback": sal_feedback
        })

        # B. Keywords (Max 30)
        kw_score = 0
        found = []
        
        # 1. Name
        if any(ent.label_ == "PERSON" for ent in self.doc.ents):
            kw_score += 4; found.append("Name")
        
        # 2. Age
        if re.search(r'\b(old|age|years)\b', text_lower):
            kw_score += 4; found.append("Age")
            
        # 3. School
        if re.search(r'\b(class|grade|school|college)\b', text_lower):
            kw_score += 4; found.append("School")
            
        # 4. Family
        if "family" in text_lower or "parents" in text_lower:
            kw_score += 4; found.append("Family")

        # 5. Hobbies
        hobby_anchors = ["I like playing", "My hobbies are", "I enjoy reading", "free time"]
        e1 = self.semantic_model.encode(hobby_anchors, convert_to_tensor=True)
        e2 = self.semantic_model.encode(self.sentences, convert_to_tensor=True)
        if util.cos_sim(e1, e2).max() > 0.35:
            kw_score += 4; found.append("Hobbies")
        
        # Bonus
        bonus = 0
        if any(x in text_lower for x in ["goal", "dream", "ambition", "explore"]): 
            bonus += 2; found.append("Goal")
        if "fun fact" in text_lower or "interesting" in text_lower:
            bonus += 2; found.append("Fun Fact")
        
        final_kw_score = min(kw_score + bonus, 30)
        
        rows.append({
            "category": "Content & Structure",
            "criteria": "Keywords Presence",
            "score": final_kw_score,
            "max": 30,
            "feedback": f"Found: {', '.join(found)}" if found else "No keywords found"
        })

        # C. Flow (Max 5)
        rows.append({
            "category": "Content & Structure",
            "criteria": "Logical Flow",
            "score": 5,
            "max": 5,
            "feedback": "Salutation → Body → Closing maintained"
        })
        
        return rows

    def _analyze_speech_rate(self):
        if not self.duration: return [{"category": "Speech Rate", "criteria": "WPM", "score": 0, "max": 10, "feedback": "No Duration"}]
        
        wpm = (self.word_count / self.duration) * 60
        score = 0
        feedback = ""

        if 111 <= wpm <= 140: 
            score = 10; feedback = "Ideal speed"
        elif 141 <= wpm <= 160: 
            score = 6; feedback = "Slightly Fast"
        elif 81 <= wpm <= 110: 
            score = 6; feedback = "Slightly Slow"
        else: 
            score = 2; feedback = "Too Fast or Too Slow"

        return [{
            "category": "Speech Rate",
            "criteria": "Words Per Minute",
            "score": score,
            "max": 10,
            "feedback": f"{int(wpm)} WPM ({feedback})"
        }]

    def _analyze_grammar_vocab(self):
        rows = []
        
        # 1. Grammar (Max 10)
        matches = self.grammar_tool.check(self.text)
        filtered = []
        for m in matches:
            rule_id = getattr(m, 'ruleId', getattr(m, 'rule_id', ''))
            if "PRP" not in rule_id: # Ignore Myself error
                filtered.append(m)
        
        err_rate = len(filtered) / self.word_count if self.word_count else 0
        g_score = 10 if err_rate < 0.03 else (8 if err_rate < 0.05 else 6)
        
        rows.append({
            "category": "Language & Grammar",
            "criteria": "Grammar Accuracy",
            "score": g_score,
            "max": 10,
            "feedback": f"{len(filtered)} issues found" if filtered else "Clean grammar"
        })

        # 2. Vocab (Max 10)
        unique = len(set([w.lower() for w in self.words]))
        ttr = unique / self.word_count if self.word_count else 0
        v_score = 10 if ttr >= 0.9 else (8 if ttr >= 0.7 else (6 if ttr >= 0.5 else 4))
        
        rows.append({
            "category": "Language & Grammar",
            "criteria": "Vocabulary (TTR)",
            "score": v_score,
            "max": 10,
            "feedback": f"Lexical diversity: {round(ttr, 2)}"
        })
        
        return rows

    def _analyze_clarity(self):
        fillers = ["um", "uh", "like", "you know", "basically"]
        count = sum(1 for w in self.words if w.lower() in fillers)
        
        score = 15 if count <= 2 else (12 if count <= 5 else 9)
        
        return [{
            "category": "Clarity",
            "criteria": "Filler Words",
            "score": score,
            "max": 15,
            "feedback": f"{count} fillers found"
        }]

    def _analyze_engagement(self):
        scores = self.sentiment_analyzer.polarity_scores(self.text)
        pos = scores['pos']
        comp = scores['compound']
        
        score = 15 if (pos >= 0.3 or comp > 0.8) else (12 if (pos >= 0.2 or comp > 0.5) else 9)
        
        return [{
            "category": "Engagement",
            "criteria": "Tone & Enthusiasm",
            "score": score,
            "max": 15,
            "feedback": "Positive and confident" if score >= 12 else "Could be more enthusiastic"
        }]