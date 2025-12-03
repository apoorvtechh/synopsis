import streamlit as st
import pandas as pd

# -------------------------
# 🌐 Page Configuration
# -------------------------
st.set_page_config(
    page_title="InsightReddit",
    page_icon="🔍",
    layout="wide"
)

# ============================================================
# 🧭 SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("🧭 Navigation")
section = st.sidebar.radio(
    "Go to section:",
    [
        "🏠 Header",
        "📂 Dataset Overview",
        "🧪 Baseline & Experiments",
        "🎯 Hyperparameter Tuning",
        "📈 DVC Pipeline Flow",
        "⚡ CI/CD Pipeline",
        "🌐 Flask API",
        "🌐 Secure API Deployment (HTTPS with Caddy & DuckDNS)",
        "🧠 Chrome Extension",
        "✨ Extension Features",
        "🧱 Tech Stack",
        "🔗 Project Repositories & Author"
    ]
)


# ============================================================
# 🏠 HEADER SECTION
# ============================================================
import streamlit as st

def section_header():
    # ============================================================
    # 🏠 Project Header
    # ============================================================
    st.markdown(
        "<h1 style='font-size:38px; color:#1E88E5;'>🔍 Reddit Sentiment Analyzer</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h3 style='color:#555;'>🚀 <b>InsightReddit</b> – Chrome Extension</h3>",
        unsafe_allow_html=True
    )

    st.write("""
    Welcome to the **Reddit Sentiment Analyzer**, a complete end-to-end DataScience and MLOps project designed to **analyze, visualize, and deploy sentiment analysis** on Reddit comments in real-time.

    👉 Use the **sidebar(Navigation)** to explore different sections.

    """)

    st.info("""
    💡 This project powers the **InsightReddit** Chrome extension, which helps users instantly gauge community sentiment directly on Reddit posts.
    """)

    # ============================================================
    # 🌐 Extension Hero Preview
    # ============================================================
    st.markdown("## 🧠 InsightReddit Extension — Live Preview")
    st.image("front.png", caption="InsightReddit Chrome Extension — Landing View", width=420)

    st.markdown(
        """
        The **InsightReddit** Chrome Extension acts as the **end-user interface**, integrating all backend components into a clean, one-click experience.  
        Once installed, users can open any Reddit post, click the extension icon, and instantly view **sentiment analysis**, **comment trends**, and **key insights** without leaving the page.
        """
    )

    # ============================================================
    # 💻 Chrome Extension Card (Add Button + Note)
    # ============================================================
    st.markdown(
        """
        <div style='
            background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
            border: 1px solid #90CAF9;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            margin: 30px 0;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        '>
            <h3 style='margin-bottom:10px; color:#0D47A1;'>🧩 Add InsightReddit to Chrome</h3>
            <p style='color:#333; font-size:16px; max-width:600px; margin:auto;'>
                Get instant <b>real-time sentiment insights</b> on any Reddit post directly in your browser — powered by ML models deployed through Flask API.
            </p>
            <a href='https://chromewebstore.google.com/detail/insightreddit/ldhjhlbadkgikjmdaknfejeoogpgpgbh?utm_source=ext_app_menu' target='_blank' style='
                display:inline-block;
                background: linear-gradient(90deg, #1976D2, #42A5F5);
                color:white;
                font-weight:bold;
                padding:12px 24px;
                border-radius:8px;
                text-decoration:none;
                margin-top:15px;
                transition: all 0.3s ease;
                box-shadow: 0px 4px 10px rgba(25,118,210,0.3);
            ' onmouseover="this.style.background='linear-gradient(90deg, #1565C0, #2196F3)'" onmouseout="this.style.background='linear-gradient(90deg, #1976D2, #42A5F5)'">
                ➕ Add Extension to Chrome
            </a>
            <p style='margin-top:15px; font-size:14px; color:#444;'>
                ⚠️ <b>Note:</b> This extension is designed to work exclusively on the <b>Google Chrome browser</b>.  
                It may not function properly on other browsers like Edge, Firefox, or Safari.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


    # ============================================================
    # ✨ Key Extension Screens (Row of 4 Images)
    # ============================================================
    st.markdown("## ✨ Key Screens of the Extension")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image("front1.png", caption="📋 Metrics Overview", use_container_width=True)

    with col2:
        st.image("pie.png", caption="🥧 Sentiment Distribution", use_container_width=True)

    with col3:
        st.image("wordcloud.png", caption="🌐 Word Clouds", use_container_width=True)

    with col4:
        st.image("sample_comment.png", caption="💬 Sample Comments", use_container_width=True)

    # ============================================================
    # 📄 Detailed Explanations for Each Image
    # ============================================================
    st.markdown("### 📝 Detailed Feature Descriptions")

    st.markdown(
        """
        - **📋 Metrics Overview Panel**  
          Provides quick, aggregated insights about the Reddit post:  
          - Total Comments analyzed  
          - Unique Commentors  
          - Average Comment Length  
          - Sentiment Score (scaled 0–10)  
          This gives users an **instant overview** of discussion dynamics.

        ---

        - **🥧 Sentiment Distribution Pie Chart**  
          Displays the proportion of **Positive**, **Neutral**, and **Negative** comments.  
          Users can **visually grasp the dominant mood** of the thread in seconds.

        ---

        - **🌐 Word Clouds (Positive / Neutral / Negative)**  
          Highlight the **most frequently used words** for each sentiment category:  
          - 🟩 Positive → joyful, supportive language  
          - 🟨 Neutral → factual/objective terms  
          - 🟥 Negative → critical or emotionally intense words  
          These clouds help **uncover hidden linguistic patterns** in the discussion.

        ---

        - **💬 Sample Comments Panel**  
          Displays **top comments** for each sentiment category.  
          Lets users explore **actual comment content** driving sentiment trends — giving both **quantitative** and **qualitative** understanding.
        """
    )



    

def section_tech_stack():
    st.header("🧱 Complete Tech Stack Used in the Project")

    st.markdown(
        """
        This project leverages a **comprehensive and modern technology stack** — from **data preprocessing & NLP** to **modeling, MLOps, deployment**, and a **Chrome extension** frontend.  
        Each layer is carefully chosen to ensure **scalability**, **reproducibility**, and **real-time performance**.
        """
    )

    # ============================================================
    # 🧠 Machine Learning & NLP
    # ============================================================
    st.subheader("🧠 Machine Learning & NLP")
    st.markdown(
        """
        - **scikit-learn** — Core ML library used for vectorization (TF–IDF), model training (Logistic Regression), and evaluation metrics.  
        - **XGBoost**, **LightGBM**, **SVM**, **KNN** — Tested as part of model selection experiments.  
        - **TextBlob** — Extracts sentiment polarity scores for handcrafted features.  
        - **NLTK** — Text preprocessing: stopword removal, tokenization, lemmatization.  
        - **Optuna** — Automated hyperparameter optimization (used for tuning Logistic Regression).  
        - **Imbalanced-learn (SMOTE)** — Handles class imbalance by synthetic oversampling.
        """
    )

    # ============================================================
    # 🧰 Data Processing & Experimentation
    # ============================================================
    st.subheader("🧰 Data Processing & Experimentation")
    st.markdown(
        """
        - **Pandas**, **NumPy** — Data cleaning, manipulation, and feature engineering.  
        - **SciPy** — Sparse matrix operations and stacking TF–IDF with handcrafted features.  
        - **DVC (Data Version Control)** — Automates and versions each ML pipeline stage (data ingestion → preprocessing → model → evaluation).  
        - **YAML** — Parameter storage for reproducible experiments.  
        - **Pickle** — Model serialization for deployment.
        """
    )

    # ============================================================
    # 🧪 MLOps & Experiment Tracking
    # ============================================================
    st.subheader("🧪 MLOps & Experiment Tracking")
    st.markdown(
        """
        - **MLflow** — For model tracking, metric logging, and model registry.  
        - **DVC Pipelines** — Automates end-to-end training workflow.  
        - **GitHub Actions** — CI/CD for running pipelines, model evaluation, and automated registration.  
        - **Model Registry (MLflow)** — Versioning and stage management (`None`, `Staging`, `Production`) for models.
        """
    )

    # ============================================================
    # 🐳 Containerization & Deployment
    # ============================================================
    st.subheader("🐳 Containerization & Deployment")
    st.markdown(
        """
        - **Flask** — Lightweight REST API for live sentiment prediction.  
        - **Docker** — Containerizes the entire application (API + model + dependencies).  
        - **Gunicorn** — WSGI server for running Flask in production.  
        - **Caddy** — Used as a **reverse proxy** with **automatic HTTPS** via Let's Encrypt.  
        - **DuckDNS** — Provides a **free dynamic DNS domain**, used to expose the API with HTTPS.  
        - **AWS EC2** — Hosted the containerized application for real-time access.  
        - **HTTPS & Domain** — Caddy + DuckDNS ensured secure, domain-based access to the API without manual certificate handling.
        """
    )

    # ============================================================
    # 🌐 Frontend & Visualization
    # ============================================================
    st.subheader("🌐 Frontend & Visualization")
    st.markdown(
        """
        - **Streamlit** — Interactive project dashboard and model reporting interface.  
        - **Matplotlib** — Visualizations like ROC curves, confusion matrices, and bar charts.  
        - **InsightReddit Chrome Extension** — Frontend layer for real-time sentiment visualization on Reddit posts.  
            - **Chart.js** — Renders sentiment pie charts.  
            - **wordcloud2.js** — Generates dynamic word clouds for each sentiment.  
            - **Vanilla JS / HTML / CSS** — Builds popup UI and handles API calls to Flask backend.
        """
    )

    # ============================================================
    # 🛠 DevOps & CI/CD
    # ============================================================
    st.subheader("🛠 DevOps & CI/CD")
    st.markdown(
        """
        - **Git & GitHub** — Version control for code, pipelines, and experiments.  
        - **GitHub Actions** — Automates pipeline runs, model registration, and API testing on every push.  
        - **DVC Remote (S3)** — Remote storage for datasets and model artifacts.  
        - **pytest** — Automated unit tests for model and API validation.  
        - **Caddy + DuckDNS** — Lightweight, automated **DevOps solution for HTTPS** and **custom domain management**, removing the need for complex reverse proxy setups.
        """
    )

    # ============================================================
    # 🐍 Languages & Environments
    # ============================================================
    st.subheader("🐍 Core Languages & Environments")
    st.markdown(
        """
        - **Python 3.11** — Primary programming language for all ML and backend components.  
        - **JavaScript** — For Chrome extension frontend logic.  
        - **HTML / CSS** — For building the extension popup interface.  
        - **Anaconda** — Virtual environment management during development.  
        - **requirements.txt** — All dependencies listed for reproducibility.
        """
    )

    st.divider()
    st.markdown(
        """
        🚀 **This full-stack integration** allowed the project to smoothly go from **raw Reddit data** → **ML pipeline** → **model registry** → **containerized API** → **secure HTTPS deployment with Caddy + DuckDNS** → **browser extension**,  
        creating a **production-ready sentiment analysis system** end to end.
        """
    )

# ============================================================
# 📂 DATASET OVERVIEW SECTION (FIXED)
# ============================================================
def section_dataset_overview():
    # -------------------------
    # 📂 Dataset Overview Section
    # -------------------------
    st.markdown("<h2 style='color:#1E88E5;'>📂 Dataset Overview</h2>", unsafe_allow_html=True)

    # 🧪 Sample data preview
    data = {
        "clean_comment": [
            "family mormon never tried explain still stare ...",
            "buddhism much lot compatible christianity espe...",
            "seriously say thing first get complex explain ...",
            "learned want teach different focus goal not wr...",
            "benefit may want read living buddha living chr..."
        ],
        "category": [0, 1, 1, -1, 1]
    }

    df_sample = pd.DataFrame(data)
    sentiment_map = {0: "Neutral", 1: "Positive", -1: "Negative"}
    df_sample["sentiment"] = df_sample["category"].map(sentiment_map)

    # ✅ FIXED: Using Markdown for bold text (no HTML mix)
    st.markdown("**Dataset Shape:** (36,793 rows × 2 columns)")
    st.dataframe(df_sample, use_container_width=True)

    # -------------------------
    # 📊 Class Imbalance (EDA Insight)
    # -------------------------
    st.markdown("<h3 style='color:#F4511E;'>📊 Class Distribution Insight</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.4])

    with col1:
        st.image(
            "imbalance_bar.png",
            caption="Class Distribution (Negative, Neutral, Positive)",
            use_container_width=True
        )

    with col2:
        st.markdown(
            """
            <div style='font-size:16px; line-height:1.7;'>
            During <b>Exploratory Data Analysis (EDA)</b>, it was observed that the dataset had a noticeable <b>class imbalance</b>:  
            <ul>
                <li>🟥 <b>Negative (-1)</b> class had approximately <b>8,000 samples</b></li>
                <li>🟨 <b>Neutral (0)</b> class had approximately <b>12,000 samples</b></li>
                <li>🟩 <b>Positive (1)</b> class had approximately <b>16,000 samples</b></li>
            </ul>

            While exploring the dataset, I observed that the distribution of sentiment classes was uneven — the negative class had around 8,000 samples, the neutral class about 12,000, and the positive class around 16,000.  

            ⚠️ Such imbalance can bias the model toward the majority class, affecting overall performance.  
            ✅ To address this and improve the model’s ability to learn from all classes fairly, <b>imbalance handling techniques</b> like SMOTE and class weighting were applied during model building.
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# 🧪 BASELINE & EXPERIMENTS SECTION
# ============================================================
def section_baseline_and_experiments():
    # 🧪 Experimentation & Model Selection
    st.markdown("<h2 style='color:#1E88E5;'>🧪 Experimentation & Model Selection</h2>", unsafe_allow_html=True)

    # ------------------------------------------------------
    # Baseline Model Overview
    # ------------------------------------------------------
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.subheader("📌 Baseline Model")
        st.markdown(
            """
The **baseline model** was trained on the **raw, imbalanced dataset without applying any sampling or advanced feature engineering**.  
The purpose of this step was to establish a **reference point** for evaluating the impact of later improvements such as vectorization, sampling techniques, and hyperparameter tuning.

Key points of this baseline:
- ❌ No sampling or imbalance handling  
- 🧾 Simple preprocessing  
- 🌱 Raw feature representation (basic setup)  
- 🎯 Acts as a benchmark to measure gains in future experiments
            """
        )
    with c2:
        st.subheader("🛠 Baseline Model — Parameters & Performance")
        colA, colB = st.columns([1, 1.4])
        with colA:
            st.markdown(
                """
                ### ⚙️ Parameters  
                The **initial baseline model** (Random Forest + CountVectorizer) was trained with:

                - 🌳 **max_depth:** 15  
                - 🌲 **n_estimators:** 200  
                - 🧮 **vectorizer_max_features:** 5000  
                - ✍️ **vectorizer_type:** CountVectorizer
                """
            )
        with colB:
            st.markdown(
                """
                ### 📈 Performance  
                - ✅ **Accuracy:** 0.65  
                - 🧪 **Weighted F1-score:** 0.57
                """
            )

    # ------------------------------------------------------
    # Vectorization Experiments
  
    st.subheader("🔬 Vectorization Experiments")
    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.image("tfidf_bow.png", caption="TF-IDF vs Bag-of-Words Experiment", use_container_width=True)
    with col2:
        st.markdown(
            """
            ### 🧠 TF-IDF vs Bag-of-Words  
            We evaluated **Bag-of-Words** vs **TF-IDF** with different n-gram ranges:

            - (1,1) → Unigrams  
            - (1,2) → Unigrams + Bigrams  
            - (1,3) → Unigrams + Bigrams + Trigrams  

            📌 Using Random Forest as the base learner,  
            **TF-IDF with (1,3)** consistently achieved **stronger macro F1** and better generalization than BOW.
            """
        )

    # ------------------------------------------------------
    # Imbalance Handling Experiments
    # ------------------------------------------------------
    st.subheader("⚖️ Imbalance Handling Experiments")
    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.image("imbalance_techq.png", caption="Imbalance Handling Techniques Comparison", use_container_width=True)
    with col2:
        st.markdown(
            """
            ### 🧠 Sampling Strategy Evaluation  
            Techniques evaluated:
            - 🔸 **Under-sampling** – Reduce majority class samples  
            - 🔸 **Over-sampling** – Duplicate minority samples  
            - 🔸 **SMOTE / SMOTEENN** – Synthetic sampling  
            - 🔸 **ADASYN** – Adaptive synthetic sampling  
            - 🔸 **Class Weights** – Loss weighting

            ✅ After several experiments, I selected **SMOTE** because it consistently provided the best **minority-class F1 score**.
            """
        )

    # ------------------------------------------------------
    # Model & Feature Experiments (Detailed)
    # ------------------------------------------------------
    st.markdown("<h3 style='color:#43A047;'>🧠 Model & Feature Experiments</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.image("knn.png", caption="KNN — Lowest Performance", use_container_width=True)
        st.image("xgboost.png", caption="XGBoost", use_container_width=True)
        st.image("svm.png", caption="SVM", use_container_width=True)
        st.image("logistic.png", caption="Logistic Regression", use_container_width=True)
        st.image("lightbgm.png", caption="LightGBM", use_container_width=True)
    with col2:
        st.markdown(
            """
            ### 🧠 What We Explored  
            In this stage, we performed a **comprehensive experimentation phase** to evaluate how different **machine learning algorithms** behave when exposed to varying **feature space sizes**.  
            The core objective was to identify a **model–vectorizer configuration** that delivers not only **high predictive performance** but also **robustness** and **efficiency** for large-scale sentiment analysis on Reddit comments.

            We systematically varied the **maximum number of features** (vocabulary size) from **1,000 to 10,000**, using TF-IDF as the base vectorizer.  
            This helped us understand how **feature dimensionality** influences both **model generalization** and **computational efficiency**.

            - 🧪 **Algorithms tested:**  
              Logistic Regression, SVM, KNN, XGBoost, LightGBM

            - ⚙️ **Feature sizes explored:** 1,000 → 10,000 (step = 1,000)  
            - 📈 For each combination, we tracked **accuracy**, **macro F1**, and **weighted F1** scores to identify stable patterns.

            ### 🏆 Key Findings  
            - ⭐ **Best feature size:** `max_features = 7000` consistently gave the **highest macro and weighted F1 scores**, indicating a **sweet spot** between capturing enough vocabulary and avoiding overfitting.  
            - 🚀 **Top performers:** **Logistic Regression** and **SVM** were clear winners. They provided **strong macro F1**, **Logistic Regression** was **computationally lightweight**, and exhibited **high stability** across different feature sizes.  
            - 🌲 **Tree-based models (XGBoost & LightGBM):** Delivered competitive scores, but required **careful hyperparameter tuning** and **more computational time**, which may be unnecessary for real-time use cases.  
            - 🧊 **KNN:** Showed **consistently weak results**, struggling in the **high-dimensional sparse space**, leading to poor generalization and slower inference.

            ### 📌 Why This Matters  
            These experiments were crucial to **narrow down the search space** for the final model.  
            By comparing models across multiple feature sizes, we avoided **premature optimization** and focused on configurations that offered the **best trade-off between accuracy, F1-score, speed, and scalability**.

            ### 📝 Takeaway  
            After extensive trials, the **TF-IDF vectorizer with 7,000 max features**, paired with a **linear model** like Logistic Regression, emerged as the **optimal choice**.  
            This combination not only achieved **top performance metrics** but also offered **faster training times**, **lighter model size**, and **easier deployment** compared to more complex alternatives.
            """
        )

    st.divider()
    st.markdown("<h2 style='color:#1E88E5;'>📌 Model Selection</h2>", unsafe_allow_html=True)

    # ------------------------------------------------------
    # Model Selection Decision
    # ------------------------------------------------------
    col3, col4 = st.columns([1, 1.4])
    with col3:
        st.image("allmodel.png", caption="🔍 Model Comparison Across Algorithms", use_container_width=True)
    with col4:
        st.markdown(
            """
            ### 🧠 Model Selection Decision  
            After evaluating the performance of all candidate algorithms across multiple feature sizes, we made a **data-driven decision** to narrow down the final model.

            - ❌ **KNN** was **removed from further consideration** due to its **consistently poor performance** in handling high-dimensional sparse TF-IDF features.  
              It struggled to generalize effectively and had longer inference times, making it unsuitable for this use case.

            - ✅ **Logistic Regression** was **selected as the final model** based on its **strong performance metrics** and **computational efficiency**.

            ### 📈 Final Model Metrics  
            - **Algorithm:** Logistic Regression  
            - **Accuracy:** 0.8721  
            - **Weighted F1-score:** 0.8716  

            These results demonstrated that Logistic Regression not only achieved **the highest scores** but also provided **stability**, **fast inference**, and **ease of deployment**, making it the ideal choice for production.
            """
        )
    st.markdown(
    """
    <div style='
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
        color: #0D1117;
        border-left: 6px solid #1565C0;
        padding: 1.2rem;
        border-radius: 12px;
        font-size: 15.5px;
        line-height: 1.7;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
    '>
        <h4 style='color:#0D47A1; margin-bottom:10px;'>🧩 Preprocessing Strategy</h4>
        <p style='margin:0;'>
            To ensure a robust and leak-free pipeline, all <b>text preprocessing</b>, 
            <b>vectorization</b> (<b>TF-IDF</b> / <b>Bag-of-Words</b>), and <b>scaling</b> operations 
            were performed <u>only after</u> the <b>train–test split</b>.
        </p>
        <br>
        <p style='margin:0;'>
            This design prevents any form of <b>data leakage</b>, ensuring that the model never "sees" 
            the test data during training. As a result, our evaluation remains <b>fair, reproducible</b>, 
            and accurately reflects <b>true model generalization</b>.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

    st.divider()
    st.markdown("<h2 style='color:#1E88E5;'>🧪 Feature Engineering</h2>", unsafe_allow_html=True)

    # ------------------------------------------------------
    # Feature Engineering Section (Detailed)
    # ------------------------------------------------------
    col5, col6 = st.columns([1.2, 1.8])
    with col5:
        st.image("final.png", caption="📊 Classification Report After Feature Engineering", use_container_width=True)
    with col6:
        st.markdown(
            """
            ### 🧠 Feature Engineering for Performance Boost  
            To further enhance model performance, we introduced **handcrafted text-based features** alongside the TF-IDF representations.  
            These additional features capture **structural** and **semantic** aspects of the text that pure vectorization might overlook.

            #### ✍️ Engineered Features:
            - 📝 **Comment Length** → Total number of characters in the comment  
            - 📄 **Word Count** → Total number of words in the comment  
            - 🔠 **Unique Word Count** → Number of unique words, indicating lexical richness  
            - 💬 **Sentiment Polarity** → Extracted using TextBlob to capture the **intrinsic sentiment** within each comment

            By combining these handcrafted features with TF-IDF vectors, we enabled the model to **leverage both statistical and linguistic signals**, resulting in improved predictive capability.

            ### 📈 Performance Improvement  
            Incorporating these features led to **significant performance gains**, as reflected in the final classification report:  
            - **Accuracy:** 0.92  
            - **Macro Avg F1-score:** 0.92  
            - **Weighted Avg F1-score:** 0.92  

            These results highlight that **carefully designed feature engineering** can provide **a substantial boost** even on top of strong vectorization techniques.
            """
        )


# ============================================================
# 🎯 HYPERPARAMETER TUNING SECTION
# ============================================================
def section_hyperparameter_tuning():
    st.markdown("<h2 style='color:#F4511E;'>🎯 Hyperparameter Tuning with Optuna</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1.8])

    with col1:
        st.markdown(
            """
            ### 🧠 Why Hyperparameter Tuning?  
            After identifying **Logistic Regression** as the best base model, the next step was to **optimize its hyperparameters** to squeeze out maximum performance.

            Instead of manually trying combinations, we used **Optuna**, an automated hyperparameter optimization framework, to efficiently explore the search space.

            Optuna performed:
            - 🔍 **Automatic search** for the best combination of `C`, `penalty`, and `solver`  
            - 🧪 **Cross-validation** to ensure robust generalization  
            - ⚡ **Pruning of bad trials** to speed up the search  
            - 📈 Selection based on **macro F1-score** as the objective metric
            """
        )

    with col2:
        st.code(
            """# ✅ Best Parameters Found using Optuna
model = LogisticRegression(
    C=1.63,
    penalty='l1',
    solver='liblinear',
    multi_class='ovr',
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)""",
            language="python"
        )

    st.markdown(
        """
        ### 🏆 Final Tuning Outcome  
        The Optuna-tuned Logistic Regression model achieved **higher macro and weighted F1 scores**, improving the model's ability to handle class imbalance while maintaining interpretability and fast inference.

        ✨ **Why this matters:**  
        - ⚖️ Balanced class weights improved minority class recall  
        - 📝 L1 penalty promoted sparsity → faster inference, less overfitting  
        - 📈 Tuned C gave a better bias-variance tradeoff

        ✅ This tuned configuration was used for **final training and registration**.
        """
    )


# ============================================================
# 📈 DVC PIPELINE FLOW SECTION
# ============================================================

def section_dvc_pipeline():
    st.markdown("<h2 style='color:#1E88E5;'>📈 DVC Pipeline Flow</h2>", unsafe_allow_html=True)

    # -----------------------------
    # DVC Overview & Image
    # -----------------------------
    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.image("dvc_flow.png", caption="DVC Pipeline Flow", width=400)

    with col2:
        st.markdown(
            """
        <div style='font-size:15px; line-height:1.6;'>
            The <b>DVC pipeline</b> defines the full end-to-end ML workflow in reproducible, version-controlled stages.  
            Each stage has:
            <ul>
                <li>🧱 <b>Inputs (deps)</b> → data, code, parameters</li>
                <li>⚙️ <b>Command (cmd)</b> → processing script</li>
                <li>📤 <b>Outputs (outs)</b> → artifacts for the next stage</li>
                <h4>📌 Pipeline Stages</h4>
                    <li>1️⃣ <b>data_ingestion</b> → Load, split raw dataset</li>
                    <li>2️⃣ <b>data_preprocessing</b> → Clean & normalize text</li>
                    <li>3️⃣ <b>model_building</b> → TF-IDF + handcrafted features + train model</li>
                    <li>4️⃣ <b>model_evaluation</b> → Evaluate model + log metrics to MLflow</li>
                    <li>5️⃣ <b>model_registration</b> → Register final model in MLflow Model Registry</li>
                
          
        </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # -----------------------------
    # dvc.yaml content
    # -----------------------------
    st.markdown("<h3 style='color:#F4511E;'>🧾 dvc.yaml — Full Pipeline Definition</h3>", unsafe_allow_html=True)

    st.code(
        """stages:
  data_ingestion:
    cmd: python src/data/data_ingestion.py
    deps:
    - src/data/data_ingestion.py
    params:
    - data_ingestion.test_size
    outs:
    - data/raw

  data_preprocessing:
    cmd: python src/data/data_preprocessing.py
    deps:
    - data/raw/train.csv
    - data/raw/test.csv
    - src/data/data_preprocessing.py
    outs:
    - data/interim

  model_building:
    cmd: python src/model/model_building.py
    deps:
    - data/interim/train_processed.csv
    - src/model/model_building.py
    params:
    - model_building.max_features
    - model_building.ngram_range
    - model_building.C
    - model_building.penalty
    - model_building.solver
    - model_building.multi_class
    - model_building.class_weight
    - model_building.max_iter
    - model_building.random_state
    - model_building.smote_random_state
    - model_building.extra_features
    - model_building.features_list
    outs:
    - logreg_model.pkl
    - tfidf_vectorizer.pkl

  model_evaluation:
    cmd: python src/model/model_evaluation.py
    deps:
    - data/interim/train_processed.csv
    - data/interim/test_processed.csv
    - src/model/model_evaluation.py
    - logreg_model.pkl
    - tfidf_vectorizer.pkl
    - params.yaml
    outs:
    - experiment_info.json

  model_registration:
    cmd: python src/model/register_model.py
    deps:
    - experiment_info.json
    - src/model/register_model.py
        """,
        language="yaml"
    )

    st.divider()

    # ============================================================
    # 🧠 TEXT PREPROCESSING STAGE
    # ============================================================
    st.markdown("<h3 style='color:#1E88E5;'>🧠 Text Preprocessing (data_preprocessing)</h3>", unsafe_allow_html=True)
    st.markdown("""
The following preprocessing steps were applied to all Reddit comments **before model training**, to ensure clean, consistent, and meaningful text input for the sentiment analysis model:

1. 🔡 **Lowercasing** – Convert all text to lowercase.  
2. ✂️ **Whitespace & newline cleanup** – Remove extra spaces and line breaks.  
3. 🧹 **Special character removal** – Keep only alphanumeric characters and key punctuation (`!?.,`).  
4. 🛑 **Stopword removal** – Remove common words but **retain** sentiment-heavy words (`not`, `no`, `but`, `yet`, `however`).  
5. 🧠 **Lemmatization** – Convert words to their base forms using WordNet Lemmatizer.  
6. 💾 **Save cleaned data** – Store processed train/test sets in `data/interim`.
""")

    st.divider()

    # ============================================================
    # 📌 MODEL BUILDING STAGE
    # ============================================================
    st.markdown("<h3 style='color:#1E88E5;'>📌 Model Building (model_building)</h3>", unsafe_allow_html=True)
    st.markdown("""
Below is the **end-to-end model building pipeline**, which transforms preprocessed text data into a trained Logistic Regression model using **TF-IDF + handcrafted features**.
""")

    st.markdown(
        """
        <div style='font-size:16px; line-height:1.7; color:#EEE;'>
        
        <b>A – load_params()</b><br>
        Loads model configuration and hyperparameters from <code>params.yaml</code>.<br><br>

        <b>B – load_data()</b><br>
        Loads preprocessed data and remaps target labels (-1→2).<br><br>

        <b>C – create_text_features()</b><br>
        Generates handcrafted numeric features to complement TF-IDF.<br>
        <u>Features created:</u> <code>comment_length, word_count, unique_word_count, num_exclamations, num_questions, sentiment polarity.</code><br><br>

        <b>D – TfidfVectorizer</b><br>
        Converts text into a TF-IDF feature matrix.<br><br>

        <b>E – Combine Features</b><br>
        Stacks TF-IDF and handcrafted features horizontally.<br><br>

        <b>F – SMOTE</b><br>
        Balances class distribution by generating synthetic samples.<br><br>

        <b>G – LogisticRegression.fit()</b><br>
        Trains Logistic Regression using tuned hyperparameters.<br><br>

        <b>H – Save with pickle</b><br>
        Saves model and vectorizer as <code>.pkl</code> for deployment.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # ============================================================
    # 🧪 MODEL EVALUATION STAGE
    # ============================================================
    st.markdown("<h3 style='color:#1E88E5;'>🧪 Model Evaluation & Registration (model_evaluation)</h3>", unsafe_allow_html=True)
    st.markdown("""
Once the model is trained, it undergoes **systematic evaluation and registration** as part of the pipeline.

This ensures that the model is not just trained — but also **tracked**, **evaluated**, and **stored** in a registry for future deployment.
""")

    st.markdown(
        """
        <div style='font-size:16px; line-height:1.7; color:#EEE;'>

        <b>1️⃣ Model Evaluation (model_evaluation.py)</b><br>
        • Loads the trained Logistic Regression model, TF-IDF vectorizer, and scaler.<br>
        • Evaluates on both <b>train</b> and <b>test</b> datasets.<br>
        • Generates <b>Precision</b>, <b>Recall</b>, <b>F1-score</b>, <b>Accuracy</b> for all 3 classes.<br>
        • Creates and logs <b>confusion matrices</b> for train and test splits.<br>
        • Logs all metrics and artifacts to <b>MLflow</b>.<br>
        • Saves run info to <code>experiment_info.json</code>.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.image("confusion_matrix_Train.png", caption="Confusion Matrix (Train)", width=400)
    with col2:
        st.image("confusion_matrix_Test.png", caption="Confusion Matrix (Test)", width=400)

    st.divider()

    # ============================================================
    # 📊 ROC & AUC SECTION
    # ============================================================
    st.markdown("<h3 style='color:#1E88E5;'>📊 ROC Curve & AUC Analysis</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧠 Train ROC Curve")
        st.image("train_roc.png", use_container_width=True)
    with col2:
        st.subheader("🧪 Test ROC Curve")
        st.image("test_roc.png", use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Train AUC**  \n- Macro: 0.9854  \n- Weighted: 0.9851")
    with col4:
        st.markdown("**Test AUC**  \n- Macro: 0.9722  \n- Weighted: 0.9725")

    st.markdown(
        """
**ROC Curve (Receiver Operating Characteristic)** shows the tradeoff between TPR and FPR.  
**AUC** measures how well the model separates classes — higher is better.

- Macro AUC treats all classes equally.  
- Weighted AUC accounts for class imbalance.

✅ Train AUC (≈0.985) and Test AUC (≈0.972) are close → **no overfitting**.
"""
    )

    st.divider()

    # ============================================================
    # 🚀 MODEL REGISTRATION STAGE
    # ============================================================
    st.markdown("<h3 style='color:#1E88E5;'>🚀 Model Registration (model_registration)</h3>", unsafe_allow_html=True)
    st.markdown(
        """
In this stage, the **trained model** is registered to the **MLflow Model Registry**, making it easy to version, manage, and deploy.

**Here's what happens in this step:**  
- 📄 Load `experiment_info.json`.  
- 🧠 Build MLflow model URI (run ID + artifact path).  
- 📝 Register model under a consistent name.  
- 🧾 Create new model version.  
- 🚦 Promote through stages (None → Staging → Production).
"""
    )

    st.image("model_registry.png", caption="MLflow Model Registry UI", use_container_width=True)

# ============================================================
# ⚡ CI/CD PIPELINE SECTION
# ============================================================
# ============================================================
# 🚀 CI/CD PIPELINE SECTION
# ============================================================
def section_cicd_pipeline():
    # --------------------------------------------------------
    # 🖼️ Display Flowchart First
    # --------------------------------------------------------
    st.markdown("<h2 style='color:#1E88E5;'>🚀 CI/CD Pipeline for MLOps</h2>", unsafe_allow_html=True)
    st.image("cicd_img2.png", caption="CI/CD + DVC + MLflow + AWS Deployment Flow", width=600)

    # --------------------------------------------------------
    # 📖 Overview
    # --------------------------------------------------------
    st.markdown(
        """
        The **CI/CD (Continuous Integration & Continuous Deployment)** pipeline is the **backbone of MLOps** in this project.  
        It ensures that every commit pushed to the `master` branch automatically:

        - 📦 Reproduces the ML pipeline (using **DVC**) if any relevant files changed  
        - 🤖 Trains and evaluates models in a consistent, automated way  
        - 📝 Registers new model versions in **MLflow Model Registry**  
        - 🧪 Runs automated model & API tests for quality control  
        - 🐳 Builds & pushes Docker images to **AWS ECR**  
        - 🌐 Deploys the updated API on **AWS EC2**

        Using **GitHub Actions** for automation, **DVC** for data/pipeline versioning, and **MLflow** for model management,  
        this pipeline achieves:

        - ✅ **Reproducibility** — Pipelines are rebuilt when code/data change.  
        - 🔄 **Automation** — Training → Registration → Testing → Deployment with no manual intervention.  
        - 🧪 **Reliability** — Automated tests block bad models.  
        - 🚀 **Faster iteration** — Every push can deploy a new working version automatically.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # 📋 Pipeline Stages
    # --------------------------------------------------------
    st.subheader("📋 CI/CD Pipeline Stages")
    st.markdown(
        """
        1. 🟦 **Checkout Code**  
           - Uses `actions/checkout@v3` to fetch the repo with full history for commit comparison.

        2. 🟨 **Set up Python**  
           - Installs Python 3.11 on the GitHub runner for running DVC and ML scripts.

        3. 🟧 **Cache & Install Dependencies**  
           - Uses `actions/cache` to reuse pip dependencies.  
           - Installs all required packages from `requirements.txt`.

        4. 🟨 **Check for Pipeline Changes (DVC)**  
           - Compares the latest commit with the previous one.  
           - If any of `dvc.yaml`, `dvc.lock`, `params.yaml`, `data/`, or `src/` changed → set `run_dvc=true`.  
           - If not → skip training & DVC to save time.

        5. 🟫 **Run DVC Pipeline (Conditional)**  
           - If `run_dvc=true`, execute `dvc repro` to run all ML stages (ingestion → preprocessing → training → evaluation).  
           - If pipeline is already up to date → skip.

        6. 🟪 **Push DVC-tracked Data to Remote**  
           - Syncs updated data & model artifacts to remote storage (e.g., AWS S3) to keep pipelines reproducible.

        7. 🟦 **Register Model in MLflow**  
           - If pipeline ran, triggers `register_model.py` to push new trained model to MLflow Model Registry automatically.

        8. 🟨 **Install & Run Model Tests**  
           - Installs `pytest`.  
           - Runs tests:
             - ✅ Model loading works  
             - 🧠 Model signature is correct  
             - 📈 Model accuracy meets minimum required threshold

        9. 🟧 **Login to AWS ECR**  
           - Authenticates the runner with AWS using secrets.  
           - Required for pushing Docker images.

        10. 🟫 **Build, Tag & Push Docker Image**  
            - Builds Docker image for the Flask API.  
            - Tags it as `latest` and pushes to your AWS ECR registry.

        11. 🟩 **Deploy to EC2**  
            - Uses `appleboy/ssh-action` to SSH into EC2.  
            - Runs `deploy.sh` to pull the new Docker image and restart the service.

        ✅ **Result:** A new model & API version is **trained, tested, containerized, and deployed** automatically after every push.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # 📝 GitHub Actions YAML Snippet
    # --------------------------------------------------------
    st.subheader("📝 GitHub Actions Workflow (cicd.yaml)")
    st.code(
        """name: CICD Pipeline

on:
  push:
    branches:
      - master

jobs:
  model-deployment:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Cache pip dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip setuptools wheel
          pip install -r requirements.txt

      - name: Check for pipeline changes
        id: check_pipeline
        run: |
          if [ -z "${{ github.event.before }}" ]; then
            echo "run_dvc=true" >> $GITHUB_ENV
            exit 0
          fi
          git fetch origin ${{ github.ref }}
          CHANGED_FILES=$(git diff --name-only ${{ github.event.before }} ${{ github.sha }})
          if echo "$CHANGED_FILES" | grep -E '(^dvc\\.yaml$|^dvc\\.lock$|^params\\.yaml$|^data/|^src/)'; then
            echo "run_dvc=true" >> $GITHUB_ENV
          else
            echo "run_dvc=false" >> $GITHUB_ENV
          fi

      - name: Run DVC pipeline
        if: env.run_dvc == 'true'
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          AWS_DEFAULT_REGION: eu-north-1
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          if dvc status -c | grep -q "Data and pipelines are up to date."; then
            echo "No DVC changes detected. Skipping repro."
          else
            dvc repro
          fi

      - name: Push DVC-tracked data
        if: env.run_dvc == 'true'
        run: dvc push

      - name: Register model in MLflow
        if: env.run_dvc == 'true'
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: python src/model/register_model.py

      - name: Install test dependencies
        run: pip install pytest

      - name: Run model tests
        if: env.run_dvc == 'true'
        run: |
          pytest scripts/test_model_loading.py
          pytest scripts/test_model_signature.py
          pytest scripts/test_model_accuracy.py

      - name: Login to AWS ECR
        run: |
          aws configure set aws_access_key_id ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws configure set aws_secret_access_key ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin 193311515551.dkr.ecr.eu-north-1.amazonaws.com

      - name: Build Docker Image
        run: docker build -t reddit_chrome_plugin:latest .

      - name: Tag Docker Image
        run: docker tag reddit_chrome_plugin:latest 193311515551.dkr.ecr.eu-north-1.amazonaws.com/reddit_chrome_plugin:latest

      - name: Push Docker Image to ECR
        run: docker push 193311515551.dkr.ecr.eu-north-1.amazonaws.com/reddit_chrome_plugin:latest

      - name: Deploy to EC2
        uses: appleboy/ssh-action@v0.1.10
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ${{ secrets.EC2_USER }}
          key: ${{ secrets.EC2_SSH_KEY }}
          script: |
            cd ~
            chmod +x deploy.sh
            ./deploy.sh
        """,
        language="yaml"
    )

# ============================================================
# 🌐 FLASK API SECTION
# ============================================================

def section_flask_api():
    st.markdown("<h2 style='color:#1E88E5;'>🌐 Reddit Sentiment Analysis API (Flask)</h2>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # 📖 Overview
    # --------------------------------------------------------
    st.markdown(
        """
        This **Flask API** acts as the **deployment layer** of the Reddit Sentiment Analysis project.  
        It exposes REST endpoints to serve real-time sentiment predictions, enabling the **Insight Reddit Chrome Extension** to display live comment sentiment directly on Reddit posts.

        🔸 The API:
        - Uses the **Reddit API** to fetch live comments from posts.  
        - Applies **the same preprocessing & feature engineering pipeline** used during model training.  
        - Loads the **latest trained model from MLflow Model Registry** for inference.  
        - Returns structured JSON responses containing sentiment labels, distributions, and metrics.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # 📡 Endpoints Table
    # --------------------------------------------------------
    st.subheader("📡 Available API Endpoints")
    st.markdown(
        """
        | Endpoint | Method | Description |
        |----------|--------|-------------|
        | `/` | **GET** | Health check — confirms that the API service is running. |
        | `/predict` | **POST** | Accepts a JSON array of comments → preprocesses → predicts sentiment → returns structured results. |
        | `/fetch/<post_id>` | **GET** | Uses the **Reddit API** to fetch comments for a given Reddit post ID, preprocesses them, predicts sentiment, and returns metrics & trends. |
        """
    )

    st.divider()

    # --------------------------------------------------------
    # 🧭 Pipeline Stages
    # --------------------------------------------------------
    st.subheader("📋 API Pipeline Stages")
    st.markdown(
        """
        1. 🧭 **Environment Setup**  
           - Load secrets (Reddit API keys, MLflow URI) from `.env`.  
           - Configure MLflow connection to retrieve the latest model.

        2. 🧠 **Model & Vectorizer Loading**  
           - Load Logistic Regression model from **MLflow Model Registry**.  
           - Load **TF–IDF vectorizer** and **scaler** for feature transformation.

        3. 📝 **Fetching Reddit Comments**  
           - **Reddit API (OAuth2)** is used to fetch comments for a given post ID.  
           - Retrieve top-level comments.  
           - Filter out deleted/removed entries.

        4. 🧼 **Preprocessing**  
           - Clean text (remove URLs, HTML, special characters).  
           - Lowercase, remove stopwords (but retain sentiment-heavy words), lemmatize.  
           - Performed **inside the API**, ensuring identical transformation as training.

        5. ✍️ **Feature Extraction**  
           - Transform cleaned text using **TF–IDF**.  
           - Generate **handcrafted features** (length, counts, sentiment polarity).  
           - Scale and combine these with TF–IDF vectors.

        6. 🤖 **Model Inference**  
           - Predict sentiment for each comment using the trained Logistic Regression model.  
           - Map numeric predictions to labels:  
             - `1 → Positive`  
             - `0 → Neutral`  
             - `-1 → Negative`

        7. 📊 **Post-processing & Metrics**  
           - Compute sentiment distribution, average scores, comment stats.  
           - Build a structured response with labeled comments and aggregate metrics.

        8. 🌐 **Respond**  
           - Return structured **JSON** to the Chrome Extension or any API client.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # 📝 Example JSON Response
    # --------------------------------------------------------
    st.subheader("📝 Example JSON Response")
    st.json({
        "percentages": {
            "positive": 55.2,
            "neutral": 27.1,
            "negative": 17.7
        },
        "metrics": {
            "total_comments": 200,
            "unique_comments": 192,
            "avg_comment_length": 120.34,
            "sentiment_score_out_of_10": 7.85
        },
        "results": [
            {"comment": "I love this!", "sentiment": "Positive", "numeric_sentiment": 1},
            {"comment": "meh, it's okay", "sentiment": "Neutral", "numeric_sentiment": 0},
            {"comment": "this is terrible", "sentiment": "Negative", "numeric_sentiment": -1}
        ]
    })

    st.divider()

    # --------------------------------------------------------
    # 🌐 Architecture Recap
    # --------------------------------------------------------
    st.subheader("🌐 API Architecture Recap")
    st.markdown(
        """
        - **Flask** serves as the inference backend.  
        - **Reddit API** supplies real-time data.  
        - **MLflow** provides the trained model & vectorizer.  
        - **Preprocessing & feature extraction** are performed within the API to ensure parity with training.  
        - The API is **deployed on EC2** and consumed by the **Chrome Extension** for real-time sentiment visualization.
        """
    )
# ============================================================
# 🌐 SECURE API DEPLOYMENT (HTTPS WITH CADDY & DUCKDNS)
# ============================================================
def section_secure_api_deployment():
    st.header("🌐 Secure API Deployment (HTTPS with Caddy & DuckDNS)")

    st.markdown(
        """
        To make the Flask API **securely accessible over the internet**,  
        I configured a **reverse proxy with Caddy** and used **DuckDNS** to enable **automatic HTTPS certificates** via Let's Encrypt.

        This secured the API endpoint and allowed the **Chrome Extension** and external clients to communicate **safely over HTTPS**.
        """
    )

    st.subheader("🧭 Why Secure the API?")
    st.markdown(
        """
        - 🔒 Prevent browser security warnings for mixed content  
        - 🌐 Make the API accessible through a **public, trusted domain**  
        - 🚀 Allow secure communication between the extension and backend  
        - 🛡️ Protect data in transit using SSL/TLS encryption
        """
    )

    st.subheader("🛠️ Tools & Services Used")
    st.markdown(
        """
        - 🌐 **DuckDNS** → Free dynamic DNS service to map the EC2 public IP to a custom domain  
        - 🧭 **Caddy** → Acts as a reverse proxy and automatically provisions HTTPS certificates  
        - 🔐 **Let's Encrypt** → Provides trusted SSL/TLS certificates
        """
    )

    st.subheader("🌐 Deployment Architecture")
    st.markdown(
        """
        The deployment flow uses **Caddy** as a front-facing secure layer that routes incoming HTTPS traffic  
        to the Flask application running on the EC2 instance:

        ```
        Client (Chrome Extension / User)
                 ↓ HTTPS
        DuckDNS Domain (e.g., insightreddit.duckdns.org)
                 ↓
              Caddy Server (Reverse Proxy + SSL)
                 ↓
          Flask API (Port 5000, EC2)
        ```
        """
    )

    st.subheader("✨ Final Outcome")
    st.markdown(
        """
        - ✅ The Flask API is now served securely over **HTTPS**  
        - 🌐 Public URL: [https://insightreddit.duckdns.org](https://insightreddit.duckdns.org)  
        - 🚀 The Chrome Extension interacts with the API without security issues  
        - 🧠 This setup improved **security**, **reliability**, and **professionalism** of the deployment.
        """
    )

    



# ============================================================
# 🧠 CHROME EXTENSION SECTION
# ============================================================
def section_chrome_extension():
    st.markdown("<h2 style='color:#1E88E5;'>🧠 InsightReddit Chrome Extension</h2>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # 📖 Overview
    # --------------------------------------------------------
    st.markdown(
        """
        The **InsightReddit** Chrome Extension is the **final user-facing layer** of the Reddit Sentiment Analysis project.  
        It provides a **real-time, on-page sentiment dashboard** for any Reddit post, powered by the backend Flask API and trained ML model.

        🔸 With a single click, users can:
        - Fetch live Reddit comments.  
        - Run them through the **Flask inference API** hosted at **[https://insightreddit.duckdns.org](https://insightreddit.duckdns.org)**.  
        - View **interactive charts**, **word clouds**, and **key sentiment metrics** directly in their browser.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # 🌐 How It Works
    # --------------------------------------------------------
    st.subheader("🌐 How InsightReddit Works")
    st.markdown(
        """
        1. 🧭 **Detect Reddit Post**  
           - When the extension is opened, it checks the active browser tab.  
           - If the user is on a Reddit post, it automatically extracts the **post ID** from the URL.

        2. 🔗 **Fetch Comments via Flask API**  
           - On clicking the **“🚀 Analyze”** button, the extension sends a GET request to:  
             ```
             https://insightreddit.duckdns.org/fetch/<post_id>
             ```
           - The Flask API internally uses the **Reddit API** to retrieve all top-level comments.  
           - These comments are **preprocessed and passed through the ML model** to get sentiment predictions.

        3. 📊 **Update Metrics and Visuals**  
           - Once the API responds, the extension updates:  
             - 📝 Total & unique comments, average length  
             - 🟢 Sentiment distribution (Positive / Neutral / Negative)  
             - 🌡 Sentiment score out of 10

        4. 🥧 **Render Charts & Word Clouds**  
           - **Chart.js** → renders a sentiment distribution pie chart.  
           - **wordcloud2.js** → generates **three separate word clouds** for positive, neutral, and negative comments.  
           - Percentage bars and top comment lists provide quick insights.

        5. 🧠 **Seamless Display**  
           - All updates happen **inside the popup UI** via DOM manipulation — no page reload or navigation required.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # 🧩 Key Components
    # --------------------------------------------------------
    st.subheader("📝 Key Components of the Extension")
    st.markdown(
        """
        - **📄 `manifest.json`**  
          - Declares required permissions (`tabs`, `activeTab`).  
          - Allows secure communication with the Flask API endpoint.  
          - Defines the extension icon and popup entry point.

        - **💻 `popup.html`**  
          - Provides the UI structure of the extension popup.  
          - Includes metric cards, sentiment charts, word clouds, and actionable buttons.

        - **🧠 `popup.js`**  
          - Core logic that:  
            - Detects the Reddit post.  
            - Sends requests to the Flask API.  
            - Processes the JSON response.  
            - Dynamically updates the DOM with metrics, charts, and top comments.

        - **📊 `Chart.js` & `wordcloud2.js`**  
          - Lightweight, locally bundled libraries used for efficient rendering of visuals.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # ⚡ User Workflow
    # --------------------------------------------------------
    st.subheader("⚡ Example User Workflow")
    st.markdown(
        """
        1. User navigates to a Reddit post.  
        2. Clicks on the **InsightReddit** extension icon — the popup opens.  
        3. The extension detects the **post ID** and displays the **“Analyze”** button.  
        4. On click, it fetches sentiment data from the **Flask API** at  
           👉 [https://insightreddit.duckdns.org](https://insightreddit.duckdns.org)  
        5. Within seconds, **charts, metrics, word clouds, and top comments** appear inside the popup.
        """
    )


# ============================================================
# ✨ EXTENSION FEATURES SECTION
# ============================================================
def section_extension_features():
    st.header("🧠 InsightReddit Chrome Extension – Features")

    st.markdown(
        """
        The **InsightReddit** Chrome Extension seamlessly connects to the Flask API to analyze Reddit post comments in **real time**.  
        It fetches comments using the **Reddit API**, applies the **same preprocessing pipeline** used during model training, and displays **rich sentiment insights directly inside the popup UI**.
        """
    )

    # -----------------------------
    # 📝 Key Feature Metrics
    # -----------------------------
    st.subheader("📊 Key Metrics Displayed in the Extension")
    st.markdown(
        """
        The extension provides **instant quantitative insights** about Reddit threads through four core metrics:

        - 💬 **Total Comments** → Shows the **total number of comments** analyzed for the current Reddit post.  
        - ✨ **Unique Commentors** → Displays how many **distinct users** contributed to the discussion, giving a sense of participation diversity.  
        - 📏 **Average Comment Length** → Indicates the **average size of comments**, which helps understand **engagement depth** (e.g., longer comments may indicate more thoughtful discussions).  
        - 🎯 **Sentiment Score (out of 10)** → Calculates an **overall sentiment score** of the thread based on polarity:  
            - Positive → contributes **1**  
            - Neutral → contributes **0.5**  
            - Negative → contributes **0**  
        A higher score indicates a more **positive community mood**, whereas lower scores suggest **neutral or negative** sentiment trends.
        """
    )

    # -----------------------------
    # 📈 Visual Insights
    # -----------------------------
    st.subheader("📈 Visual Insights in the Extension")
    st.markdown(
        """
        Along with numerical metrics, the extension provides **powerful visualizations** to quickly understand sentiment distribution and language patterns:

        - 🥧 **Sentiment Distribution Pie Chart**  
            - Displays the **proportion of positive, neutral, and negative comments** for the post.  
            - Helps identify the **overall emotional tone** at a glance.  
            - Rendered dynamically using **Chart.js**.

        - 🌈 **Word Clouds for Each Sentiment**  
            - 😃 **Positive Word Cloud** → Highlights **frequently used positive expressions**, giving insight into what users appreciate.  
            - 😐 **Neutral Word Cloud** → Shows **neutral/common words**, reflecting the general context of discussions.  
            - 😡 **Negative Word Cloud** → Emphasizes **negative keywords**, making it easy to spot pain points or complaints.  
            - Generated in real time using `wordcloud2.js`.

        - 💬 **Top Comments by Sentiment**  
            - Lists the **top 5 comments** from each sentiment category (Positive / Neutral / Negative).  
            - Gives **qualitative context** behind the distribution, helping understand why users feel a certain way.
        """
    )

    st.info("✅ All visualizations are dynamically updated based on the live Reddit data fetched for each post, giving users **instant sentiment insights** without leaving the page.")
def section_repos():
    st.title("🔗 Project Repositories & Author")

    st.write(
        """
        Below are all repositories associated with this project —
        the **main ML repo**, **Chrome extension**, and the **experimentation repo**.
        """
    )

    # ---------- Repos (NO indentation inside the HTML string!) ----------
    st.markdown(
"""
<div style="background-color:#f1f5ff;padding:22px;border-radius:14px;
border:1px solid #c5d4ff;box-shadow:0 2px 8px rgba(0,0,0,0.06);margin-top:12px;">

<h3 style="color:#0b2e72; margin-bottom:12px;">📦 Project Repositories</h3>

<ul style="color:#111; font-size:16px; line-height:1.7;">

<li>
<b>Main Project Repository:</b><br>
<a href="https://github.com/apoorvtechh/reddit-comment-sentiment-analysis" target="_blank">
github.com/apoorvtechh/reddit-comment-sentiment-analysis
</a>
</li>

<li style="margin-top:14px;">
<b>Chrome Extension Repository:</b><br>
<a href="https://github.com/apoorvtechh/reddit-yt-plugin" target="_blank">
github.com/apoorvtechh/reddit-yt-plugin
</a>
</li>

<li style="margin-top:14px;">
<b>Experimentation Repository:</b><br>
<a href="https://github.com/apoorvtechh/Second_project" target="_blank">
github.com/apoorvtechh/Second_project
</a>
</li>

</ul>

</div>
""",
        unsafe_allow_html=True,
    )

    # ---------- Author ----------
    st.markdown(
"""
<div style="background-color:#fff8e8;padding:22px;border-radius:14px;
border:1px solid #ffddb3;box-shadow:0 2px 8px rgba(0,0,0,0.06);margin-top:18px;">

<h3 style="color:#9b5300; margin-bottom:10px;">👨‍💻 Author</h3>

<p style="color:#222; font-size:16px; line-height:1.55;">
<b>Apoorv Gupta</b><br>
Email: <a href="mailto:apoorvtecgg@gmail.com">apoorvtecgg@gmail.com</a><br>
GitHub: <a href="https://github.com/apoorvtechh" target="_blank">
github.com/apoorvtechh
</a>
</p>

</div>
""",
        unsafe_allow_html=True,
    )




# ============================================================
# ✅ SECTION NAVIGATION LOGIC (FINAL DISPATCH)
# ============================================================
if section == "🏠 Header":
    section_header()
elif section == "📂 Dataset Overview":
    section_dataset_overview()
elif section == "🧪 Baseline & Experiments":
    section_baseline_and_experiments()
elif section == "🎯 Hyperparameter Tuning":
    section_hyperparameter_tuning()
elif section == "📈 DVC Pipeline Flow":
    section_dvc_pipeline()
elif section == "⚡ CI/CD Pipeline":
    section_cicd_pipeline()
elif section == "🌐 Flask API":
    section_flask_api()
elif section == "🌐 Secure API Deployment (HTTPS with Caddy & DuckDNS)":
    section_secure_api_deployment()
elif section == "🧠 Chrome Extension":
    section_chrome_extension()
elif section == "✨ Extension Features":
    section_extension_features()
elif section == "🧱 Tech Stack":
    section_tech_stack()
elif section == "🔗 Project Repositories & Author":
    section_repos()