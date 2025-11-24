import streamlit as st
import pandas as pd
import spacy
import language_tool_python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from scorer import ScoringEngine 

# --- CONFIGURATION & CACHING ---
st.set_page_config(page_title="AI Speech Scorer", layout="wide")

@st.cache_resource
def load_models():
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2') 
    grammar_tool = language_tool_python.LanguageTool('en-US')
    sentiment_analyzer = SentimentIntensityAnalyzer()
    
    return nlp, semantic_model, grammar_tool, sentiment_analyzer

with st.spinner("Loading AI Models..."):
    nlp, semantic_model, grammar_tool, sentiment_analyzer = load_models()

def main():
    st.title("🎙️ AI Speech Scorer")
    st.markdown("### Automated Rubric-Based Assessment")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        transcript_input = st.text_area("Paste Transcript Here:", height=300, placeholder="Enter text...")
    with col2:
        duration_input = st.number_input("Audio Duration (seconds):", min_value=0, value=60)
        analyze_btn = st.button("Analyze Transcript", type="primary")

    if analyze_btn:
        if duration_input <= 0:
            st.error("⚠️ Error: Duration must be > 0"); st.stop()
        if not transcript_input.strip():
            st.error("⚠️ Error: Transcript empty"); st.stop()

        with st.spinner("Analyzing..."):
            engine = ScoringEngine(transcript_input, duration_input, nlp, semantic_model, grammar_tool, sentiment_analyzer)
            results = engine.evaluate()
            
            st.divider()
            
            # 1. Overall Score Banner
            score_val = results['overall_score']
            if score_val >= 80:
                st.success(f"🌟 **Excellent! Overall Score: {score_val}/100**")
            elif score_val >= 60:
                st.warning(f"👍 **Good Effort! Overall Score: {score_val}/100**")
            else:
                st.error(f"📉 **Needs Improvement. Overall Score: {score_val}/100**")
            
            # Meta Stats
            st.caption(f"**Word Count:** {results['meta']['word_count']} words | **Speech Rate:** {results['meta']['wpm']} WPM")
            st.write("")

            st.subheader("Score Breakdown")
            st.info("👇 Click on a category below to see the detailed criteria and feedback.")

            # 2. Interactive Expanders (The "Dropdown" View)
            # Convert data to DataFrame for easier filtering
            df_all = pd.DataFrame(results['table_data'])
            
            # Define the logical order of categories
            categories = [
                "Content & Structure", 
                "Speech Rate", 
                "Language & Grammar", 
                "Clarity", 
                "Engagement"
            ]
            
            for cat in categories:
                # Get data for this category
                cat_data = df_all[df_all['category'] == cat]
                
                if cat_data.empty: continue
                
                # Calculate Totals for the header
                total_score = cat_data['score'].sum()
                max_score = cat_data['max'].sum()
                
                # Create the clickable Header label
                # Removed visual indicators (emojis) and added Weightage info
                label = f"**{cat}** — Score: **{int(total_score)} / {int(max_score)}** (Weightage: {int(max_score)}%)"
                
                with st.expander(label, expanded=False):
                    # Prepare clean table for inside the dropdown
                    display_df = cat_data[['criteria', 'score', 'max', 'feedback']].copy()
                    display_df.columns = ['Criteria', 'Score', 'Max', 'Feedback']
                    
                    st.dataframe(
                        display_df,
                        column_config={
                            "Criteria": st.column_config.TextColumn("Criteria", width="medium"),
                            "Score": st.column_config.NumberColumn("Score", format="%d"),
                            "Max": st.column_config.NumberColumn("Max", format="%d"),
                            "Feedback": st.column_config.TextColumn("AI Feedback", width="large"),
                        },
                        hide_index=True,
                        use_container_width=True
                    )

if __name__ == "__main__":
    main()