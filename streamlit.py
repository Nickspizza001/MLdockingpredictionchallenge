import streamlit as st
import pandas as pd
import sqlite3
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np




# --- DATABASE SETUP ---
conn = sqlite3.connect("leaderboard.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS leaderboard (
    email TEXT PRIMARY KEY,
    mae REAL,
    rmse REAL,
    r2 REAL,
    ef10 REAL,
    submission_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# --- GROUND TRUTH ---
GROUND_TRUTH_FILE = "docked_external_cleaned.csv"
ground_truth = pd.read_csv(GROUND_TRUTH_FILE)

# --- PAGE NAVIGATION ---
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["📤 Submit Prediction", "📊 View Leaderboard"])

# ================================
# 📤 SUBMISSION PAGE
# ================================
if page == "📤 Submit Prediction":
    st.title("`Docking Score Prediction Challenge`: Chemoinformatics Academy")
    st.markdown("""
### Background and Task

Many studies have shown that scoring functions are important in drug discovery. But one big problem is the balance between accuracy and how much time and computer power it takes. Tools like molecular dynamics (MD) and FEP+ give results that are close to experimental values, but they are very slow and expensive to run. This is why machine learning (ML) models are now being used to help speed things up [Warren et al., 2024](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/6675a38d5101a2ffa8274f62/original/how-to-make-machine-learning-scoring-functions-competitive-with-fep.pdf) .

In this small project, we want to show how ML can help reduce the time and effort needed in virtual screening. While protein-ligand interactions and binding poses are very important in drug discovery, our goal here is to help participants learn how ML can be useful.

We have docking scores from XP (extra precision) docking for ~1,000 compounds. Running this docking took more than a day. Your job is to use this data to build a model that can predict the docking scores of 286 new compounds. These 286 compounds were also docked using XP mode, and that process took about half a day.

This test set contains a mix of known active and inactive compounds for a VEGFR target. Your goal is:

- To predict the XP docking scores for these 286 compounds.
- To see if your model can help find the active compounds from the mix.

This project is a simple way to show how ML can help reduce the time and cost of drug discovery. 
                
### Submission Guidelines
- Please submit a CSV file with two columns: `ChemblID` and your predicted docking scores.
- Any issue you face, Please send an email to chemoinfomaticsacademy@gmail.com
                
#### Note
- You can watch this Drug Hunter Episode [Practical Applications of Physics-based Modeling for Medicinal Chemists](https://www.youtube.com/watch?v=7xXIMfgn3h8&t=2761s) and follow the Drug Hunter Channel on YouTube
""")

    st.subheader("📂 Download Datasets")

    # Load datasets
    training_test_data = pd.read_csv("ML_challenge_training_dataset.csv")  # Example training data
    external_data_to_predict = pd.read_csv("ML_challenge_external_dataset.csv")      # Your actual test data

    # Convert to CSV
    train_csv = training_test_data.to_csv(index=False).encode("utf-8")
    test_csv = external_data_to_predict.to_csv(index=False).encode("utf-8")

    # Download buttons
    st.download_button(
        label="📥 Download Training Dataset",
        data=train_csv,
        file_name="ML_challenge_training_dataset.csv",
        mime="text/csv"
    )

    st.download_button(
        label="📥 Download External Dataset to predict",
        data=test_csv,
        file_name="ML_challenge_external_dataset.csv",
        mime="text/csv"
    )


    with st.form("upload_form"):
        email = st.text_input("📧 Enter your email")
        target_col = st.text_input("🎯 Column name for predicted docking scores (e.g., 'docking score')")
        uploaded_file = st.file_uploader("📄 Upload your prediction CSV file", type="csv")
        submit = st.form_submit_button("Submit")

    if submit and uploaded_file is not None and email and target_col:
        try:
            submission = pd.read_csv(uploaded_file)

            # Validate required columns
            if not {"ChemblID", target_col}.issubset(submission.columns):
                st.error(f"❌ Your file must contain 'ChemblID' and '{target_col}' columns.")
            else:
                # Validate ordering
                submitted_ids = list(submission["ChemblID"])
                expected_ids = list(ground_truth["ChemblID"])

                if submitted_ids != expected_ids:
                    st.error("❌ The order of ChemblIDs in your file does not match the ground truth.")
                else:
                    # Compute metrics
                    y_true = ground_truth["docking score"]
                    y_pred = submission[target_col]

                    mae = mean_absolute_error(y_true, y_pred)
                    mse = mean_squared_error(y_true, y_pred)
                    rmse = rmse = np.sqrt(mse)
                    r2 = r2_score(y_true, y_pred)

                    # Enrichment Factor @ 10%
                    merged = ground_truth.copy()
                    merged["DockingScore_Pred"] = y_pred
                    top_10pct = int(len(merged) * 0.10)
                    merged_sorted = merged.sort_values("DockingScore_Pred")  # Lower is better
                    top_hits = merged_sorted.head(top_10pct)

                    active_cutoff = 7.0  # or 8.0 depending on your use case
                    n_actives_total = (merged["pChEMBL Value"] >= active_cutoff).sum()
                    n_actives_top = (top_hits["pChEMBL Value"] >= active_cutoff).sum()

                    expected_random = top_10pct * (n_actives_total / len(merged)) if len(merged) > 0 else 1
                    ef10 = n_actives_top / expected_random if expected_random > 0 else 0

                    # --- STORE/UPDATE DB ---
                    cursor.execute("SELECT ef10 FROM leaderboard WHERE email = ?", (email,))
                    existing = cursor.fetchone()

                    if existing and ef10 >= existing[0]:
                        st.warning(f"Your EF10 ({ef10:.2f}) is not better than your previous score ({existing[0]:.2f}). Not updated.")
                    else:
                        if existing:
                            cursor.execute("""
                                UPDATE leaderboard
                                SET mae=?, rmse=?, r2=?, ef10=?, submission_time=CURRENT_TIMESTAMP
                                WHERE email=?
                            """, (mae, rmse, r2, ef10, email))
                            st.success(f"🎉 Improved! Updated with better score.")
                        else:
                            cursor.execute("""
                                INSERT INTO leaderboard (email, mae, rmse, r2, ef10)
                                VALUES (?, ?, ?, ?, ?)
                            """, (email, mae, rmse, r2, ef10))
                            st.success(f"✅ Submission recorded.")

                        conn.commit()

                    # --- SCATTER PLOT ---
                    fig, ax = plt.subplots()
                    ax.scatter(y_true, y_pred, color="blue", alpha=0.6)
                    ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Perfect Prediction")
                    ax.set_xlabel("True Docking Score")
                    ax.set_ylabel("Predicted Docking Score")
                    ax.set_title("📈 True vs Predicted Docking Scores")
                    ax.legend()
                    st.pyplot(fig)

                    # --- METRICS DISPLAY ---
                    st.metric("MAE", f"{mae:.4f}")
                    st.metric("RMSE", f"{rmse:.4f}")
                    st.metric("R²", f"{r2:.4f}")
                    st.metric("EF@10%", f"{ef10:.2f}")

        except Exception as e:
            st.error(f"❌ Failed to process submission: {e}")

# ================================
# 📊 LEADERBOARD PAGE
# ================================
elif page == "📊 View Leaderboard":
    st.title("🏆 Leaderboard (Ranked by EF@10%)")

    leaderboard_df = pd.read_sql_query(
        "SELECT email, mae, rmse, r2, ef10, submission_time FROM leaderboard ORDER BY ef10 DESC",
        conn
    )

    # Add rank column
    leaderboard_df.insert(0, "Rank", range(1, len(leaderboard_df) + 1))

    # --- PAGINATION ---
    page_size = 10
    total_pages = (len(leaderboard_df) - 1) // page_size + 1
    page_num = st.number_input("Page", 1, total_pages, step=1)

    start_idx = (page_num - 1) * page_size
    end_idx = start_idx + page_size

    st.dataframe(leaderboard_df.iloc[start_idx:end_idx], use_container_width=True)

    # --- DOWNLOAD CSV ---
    csv = leaderboard_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Full Leaderboard as CSV", csv, "leaderboard.csv", "text/csv")
