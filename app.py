import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Hotel Review Intelligence",
    page_icon="🏨",
    layout="wide"
)


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================================================
# HOTEL ASPECTS
# =========================================================

aspects = {
    "Room": [
        "room",
        "rooms",
        "bed",
        "bathroom"
    ],

    "Cleanliness": [
        "clean",
        "dirty",
        "cleanliness",
        "hygiene"
    ],

    "Staff": [
        "staff",
        "service",
        "reception",
        "employee"
    ],

    "Food": [
        "food",
        "breakfast",
        "restaurant",
        "dinner"
    ],

    "Location": [
        "location",
        "area",
        "place",
        "near"
    ],

    "Wi-Fi": [
        "wifi",
        "wi-fi",
        "internet"
    ],

    "Facilities": [
        "pool",
        "gym",
        "parking",
        "facilities"
    ]
}


# =========================================================
# LOAD + PREPROCESS DATASET
# Cached so Streamlit does not reload/process 20K reviews
# on every interaction.
# =========================================================

@st.cache_data
def load_dataset():
    data = pd.read_csv(
        "data/b.csv",
        encoding="latin1"
    )

    data["Clean_Review"] = (
        data["Review"]
        .astype(str)
        .apply(clean_text)
    )

    data["Sentiment"] = data["Rating"].apply(
        lambda rating:
        "Negative" if rating <= 2
        else "Neutral" if rating == 3
        else "Positive"
    )

    return data


df = load_dataset()


# =========================================================
# SENTIMENT COUNTS
# =========================================================

positive = (df["Rating"] >= 4).sum()
neutral = (df["Rating"] == 3).sum()
negative = (df["Rating"] <= 2).sum()


# =========================================================
# TITLE
# =========================================================

st.title("🏨 Hotel Review Intelligence System")

st.subheader(
    "Turn Guest Reviews into Actionable Insights"
)

st.write(
    "Welcome to the Hotel Manager Dashboard"
)


# =========================================================
# REVIEW OVERVIEW
# =========================================================

st.markdown("### 📊 Review Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Reviews",
    len(df)
)

col2.metric(
    "Positive Reviews",
    positive
)

col3.metric(
    "Neutral Reviews",
    neutral
)

col4.metric(
    "Negative Reviews",
    negative
)


# =========================================================
# SENTIMENT OVERVIEW
# =========================================================

st.markdown("### 😊 Sentiment Overview")

sentiment_data = pd.DataFrame({
    "Sentiment": [
        "Positive",
        "Neutral",
        "Negative"
    ],
    "Reviews": [
        positive,
        neutral,
        negative
    ]
})

st.bar_chart(
    sentiment_data.set_index("Sentiment")
)


# =========================================================
# PRECOMPUTED ASPECT MASKS
# Cached so keyword scanning is not repeated on every rerun.
# =========================================================

@st.cache_data
def calculate_aspect_data(reviews):
    review_series = pd.Series(reviews)

    aspect_masks = {}
    aspect_results = []

    for aspect, keywords in aspects.items():
        mask = review_series.str.lower().apply(
            lambda review: any(
                keyword in review
                for keyword in keywords
            )
        )

        aspect_masks[aspect] = mask.tolist()

        aspect_results.append({
            "Aspect": aspect,
            "Reviews": int(mask.sum())
        })

    return aspect_results, aspect_masks


aspect_results, aspect_masks = calculate_aspect_data(
    tuple(df["Review"].astype(str))
)


# =========================================================
# ASPECT ANALYSIS
# =========================================================

st.markdown("### 🏨 Aspect Analysis")

aspect_df = pd.DataFrame(aspect_results)

st.bar_chart(
    aspect_df.set_index("Aspect")
)


# =========================================================
# MANAGER ACTION CENTER
# =========================================================

st.markdown("### 🚨 Manager Action Center")

recommendations = []

for aspect in aspects:

    aspect_mask = pd.Series(
        aspect_masks[aspect],
        index=df.index
    )

    ratings = df.loc[
        aspect_mask,
        "Rating"
    ]

    if len(ratings) > 0:

        negative_count = (
            ratings <= 2
        ).sum()

        total = len(ratings)

        negative_percentage = (
            negative_count / total
        ) * 100

        if negative_percentage >= 15:
            priority = "🔴 HIGH"
            action = "Immediate improvement required"

        elif negative_percentage >= 10:
            priority = "🟡 MEDIUM"
            action = "Monitor and improve"

        else:
            priority = "🟢 LOW"
            action = "Maintain current performance"

        recommendations.append({
            "Aspect": aspect,
            "Negative %": round(
                negative_percentage,
                2
            ),
            "Priority": priority,
            "Recommended Action": action
        })


recommendation_df = pd.DataFrame(
    recommendations
)

st.dataframe(
    recommendation_df,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# TOP NEGATIVE REVIEW TERMS
# Cached because this calculation is expensive.
# =========================================================

@st.cache_data
def get_negative_terms(reviews):

    negative_reviews = pd.Series(reviews).astype(str)

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_features=15
    )

    term_matrix = vectorizer.fit_transform(
        negative_reviews
    )

    term_counts = term_matrix.sum(
        axis=0
    ).A1

    terms = vectorizer.get_feature_names_out()

    return pd.DataFrame({
        "Term": terms,
        "Mentions": term_counts
    }).sort_values(
        by="Mentions",
        ascending=False
    )


negative_terms_df = get_negative_terms(
    tuple(
        df.loc[
            df["Rating"] <= 2,
            "Review"
        ].astype(str)
    )
)

st.markdown(
    "### ⚠️ Top Negative Review Terms"
)

st.bar_chart(
    negative_terms_df.set_index("Term")
)


# =========================================================
# REVIEW EXPLORER
# =========================================================

st.markdown("### 🔎 Review Explorer")

search_text = st.text_input(
    "Search reviews",
    placeholder=(
        "Example: room, breakfast, staff, wifi..."
    ),
    key="review_search"
)

if search_text:

    results = df[
        df["Review"]
        .astype(str)
        .str.contains(
            search_text,
            case=False,
            na=False
        )
    ]

    st.write(
        f"Found {len(results)} matching reviews"
    )

    st.dataframe(
        results[
            ["Review", "Rating"]
        ].head(20),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# POSITIVE REVIEW INSIGHTS
# Cached because this calculation is expensive.
# =========================================================

@st.cache_data
def get_positive_terms(reviews):

    positive_reviews = pd.Series(
        reviews
    ).astype(str)

    positive_vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_features=15
    )

    positive_matrix = (
        positive_vectorizer.fit_transform(
            positive_reviews
        )
    )

    positive_counts = (
        positive_matrix.sum(
            axis=0
        ).A1
    )

    positive_terms = (
        positive_vectorizer
        .get_feature_names_out()
    )

    return pd.DataFrame({
        "Term": positive_terms,
        "Mentions": positive_counts
    }).sort_values(
        by="Mentions",
        ascending=False
    )


positive_terms_df = get_positive_terms(
    tuple(
        df.loc[
            df["Rating"] >= 4,
            "Review"
        ].astype(str)
    )
)

st.markdown(
    "### 🌟 What Guests Like"
)

st.bar_chart(
    positive_terms_df.set_index("Term")
)


# =========================================================
# RATING DISTRIBUTION
# =========================================================

st.markdown(
    "### ⭐ Rating Distribution"
)

rating_counts = (
    df["Rating"]
    .value_counts()
    .sort_index()
)

st.bar_chart(
    rating_counts
)


# =========================================================
# DOWNLOAD MANAGER REPORT
# =========================================================

st.markdown(
    "### 📥 Download Manager Report"
)

report_csv = (
    recommendation_df
    .to_csv(index=False)
)

st.download_button(
    label="Download Manager Report",
    data=report_csv,
    file_name="hotel_manager_report.csv",
    mime="text/csv"
)


# =========================================================
# TRAIN NLP MODEL FOR NEW REVIEWS
# IMPORTANT:
# The model is trained ONCE and cached.
# New review prediction is therefore much faster.
# =========================================================

@st.cache_resource
def train_review_model(clean_reviews, sentiments):

    review_tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2
    )

    X = review_tfidf.fit_transform(
        clean_reviews
    )

    review_model = LogisticRegression(
        max_iter=1000
    )

    review_model.fit(
        X,
        sentiments
    )

    return review_tfidf, review_model, X


review_tfidf, review_model, X = train_review_model(
    tuple(df["Clean_Review"]),
    tuple(df["Sentiment"])
)


# =========================================================
# NEW REVIEW INPUT
# =========================================================

st.markdown(
    "### 🧠 Analyze a New Hotel Review"
)

new_review = st.text_area(
    "Enter a customer review:",
    placeholder=(
        "Example: The room was clean "
        "and the staff were very helpful."
    ),
    key="new_review_input"
)


# =========================================================
# ANALYZE NEW REVIEW
# =========================================================

if st.button(
    "Analyze Review",
    key="analyze_review_button"
):

    if new_review.strip():

        # Clean new review
        cleaned_review = clean_text(
            new_review
        )

        # Convert review into TF-IDF
        new_review_vector = (
            review_tfidf.transform(
                [cleaned_review]
            )
        )

        # Predict sentiment
        prediction = (
            review_model.predict(
                new_review_vector
            )[0]
        )

        # =================================================
        # SENTIMENT RESULT
        # =================================================

        st.markdown(
            "### 📌 Prediction Result"
        )

        if prediction == "Positive":

            st.success(
                "😊 Sentiment: POSITIVE"
            )

        elif prediction == "Negative":

            st.error(
                "😞 Sentiment: NEGATIVE"
            )

        else:

            st.warning(
                "😐 Sentiment: NEUTRAL"
            )

        # =================================================
        # ASPECT DETECTION
        # =================================================

        detected_aspects = []

        review_lower = cleaned_review.lower()

        for aspect, keywords in aspects.items():

            for keyword in keywords:

                if keyword in review_lower:

                    detected_aspects.append(
                        aspect
                    )

                    break

        # =================================================
        # DISPLAY DETECTED ASPECTS
        # =================================================

        st.markdown(
            "### 🏨 Detected Aspects"
        )

        if detected_aspects:

            for aspect in detected_aspects:

                st.info(
                    f"🔹 {aspect}"
                )

        else:

            st.info(
                "No specific hotel aspect detected."
            )

        # =================================================
        # MANAGER ACTION
        # =================================================

        st.markdown(
            "### 🚨 Manager Action"
        )

        if prediction == "Negative":

            if detected_aspects:

                aspect_text = ", ".join(
                    detected_aspects
                )

                st.error(
                    f"Immediate attention required for: "
                    f"{aspect_text}"
                )

            else:

                st.error(
                    "Review is negative. "
                    "Manager should investigate the issue."
                )

        elif prediction == "Positive":

            if detected_aspects:

                aspect_text = ", ".join(
                    detected_aspects
                )

                st.success(
                    f"Maintain good performance in: "
                    f"{aspect_text}"
                )

            else:

                st.success(
                    "Customer is satisfied. "
                    "Maintain current service quality."
                )

        else:

            if detected_aspects:

                aspect_text = ", ".join(
                    detected_aspects
                )

                st.warning(
                    f"Monitor customer feedback related to: "
                    f"{aspect_text}"
                )

            else:

                st.warning(
                    "Monitor this review for possible improvement areas."
                )

    else:

        st.warning(
            "Please enter a review."
        )


# =========================================================
# HOTEL HEALTH SCORE
# =========================================================

st.markdown("---")
st.markdown("## 🏥 Hotel Health Score")

positive_percentage = (
    positive / len(df)
) * 100

negative_percentage = (
    negative / len(df)
) * 100

average_rating = df["Rating"].mean()

rating_score = (
    average_rating / 5
) * 100

health_score = (
    (positive_percentage * 0.5)
    + (rating_score * 0.3)
    + ((100 - negative_percentage) * 0.2)
)

health_score = round(
    health_score,
    1
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "🏥 Hotel Health Score",
    f"{health_score}/100"
)

col2.metric(
    "⭐ Average Rating",
    f"{average_rating:.2f}/5"
)

col3.metric(
    "😊 Positive Feedback",
    f"{positive_percentage:.1f}%"
)

if health_score >= 80:

    st.success(
        "🟢 Excellent Hotel Health — Guests are highly satisfied."
    )

elif health_score >= 60:

    st.warning(
        "🟡 Good Hotel Health — Some areas need improvement."
    )

else:

    st.error(
        "🔴 Hotel Health Needs Attention — Major improvements are recommended."
    )


# =========================================================
# HOTEL STRENGTH vs WEAKNESS MAP
# =========================================================

st.markdown("---")
st.markdown("## 🏆 Hotel Strength vs Weakness Map")

strength_results = []

for aspect in aspects:

    aspect_mask = pd.Series(
        aspect_masks[aspect],
        index=df.index
    )

    aspect_ratings = df.loc[
        aspect_mask,
        "Rating"
    ]

    if len(aspect_ratings) > 0:

        average_aspect_rating = (
            aspect_ratings.mean()
        )

        if average_aspect_rating >= 4.0:
            status = "🟢 Strong"

        elif average_aspect_rating >= 3.5:
            status = "🔵 Good"

        elif average_aspect_rating >= 3.0:
            status = "🟡 Needs Improvement"

        else:
            status = "🔴 Weak"

        strength_results.append({
            "Aspect": aspect,
            "Average Rating": round(
                average_aspect_rating,
                2
            ),
            "Performance": status,
            "Reviews Analyzed": len(
                aspect_ratings
            )
        })


strength_df = pd.DataFrame(
    strength_results
)

strength_df = strength_df.sort_values(
    by="Average Rating",
    ascending=False
)

st.dataframe(
    strength_df,
    use_container_width=True,
    hide_index=True
)

st.markdown("### 📊 Aspect Performance")

st.bar_chart(
    strength_df.set_index("Aspect")[
        "Average Rating"
    ]
)

strong_aspects = strength_df[
    strength_df["Average Rating"] >= 4.0
]["Aspect"].tolist()

weak_aspects = strength_df[
    strength_df["Average Rating"] < 3.0
]["Aspect"].tolist()

if strong_aspects:

    st.success(
        "🏆 Hotel Strengths: " +
        ", ".join(strong_aspects)
    )

if weak_aspects:

    st.error(
        "⚠️ Hotel Weaknesses: " +
        ", ".join(weak_aspects)
    )

if not weak_aspects:

    st.info(
        "✅ No aspect is currently classified as Weak."
    )


# =========================================================
# SIMILAR COMPLAINT DETECTION
# =========================================================

st.markdown("---")
st.markdown("## 🔍 Similar Complaint Detection")

st.write(
    "Enter a customer complaint to find similar complaints "
    "from existing hotel reviews."
)

similar_complaint = st.text_area(
    "📝 Enter Customer Complaint",
    placeholder=(
        "Example: The breakfast was cold "
        "and the food quality was poor."
    ),
    key="similar_complaint_input"
)

if st.button(
    "🔍 Find Similar Complaints",
    key="find_similar_button"
):

    if similar_complaint.strip() == "":

        st.warning(
            "⚠️ Please enter a complaint first."
        )

    else:

        cleaned_complaint = clean_text(
            similar_complaint
        )

        new_complaint_vector = (
            review_tfidf.transform(
                [cleaned_complaint]
            )
        )

        negative_indices = (
            df.index[
                df["Rating"] <= 2
            ].tolist()
        )

        negative_vectors = X[
            negative_indices
        ]

        similarity_scores = cosine_similarity(
            new_complaint_vector,
            negative_vectors
        ).flatten()

        top_positions = (
            similarity_scores
            .argsort()[-5:][::-1]
        )

        similar_results = []

        for position in top_positions:

            original_index = (
                negative_indices[position]
            )

            similarity_percentage = (
                similarity_scores[position]
                * 100
            )

            similar_results.append({
                "Complaint": df.loc[
                    original_index,
                    "Review"
                ],
                "Rating": df.loc[
                    original_index,
                    "Rating"
                ],
                "Similarity": (
                    f"{similarity_percentage:.1f}%"
                )
            })

        similar_results_df = pd.DataFrame(
            similar_results
        )

        st.markdown(
            "### 📋 Most Similar Complaints"
        )

        st.dataframe(
            similar_results_df,
            use_container_width=True,
            hide_index=True
        )

        highest_similarity = (
            similarity_scores[
                top_positions[0]
            ]
        )

        if highest_similarity >= 0.70:

            st.error(
                "🚨 Strong repeated complaint detected! "
                "This complaint is highly similar to previous negative reviews."
            )

        elif highest_similarity >= 0.50:

            st.warning(
                "⚠️ Similar complaint pattern detected. "
                "The hotel manager should investigate this issue."
            )

        else:

            st.success(
                "✅ No strong repeated complaint pattern detected."
            )

        detected_aspects = []

        complaint_lower = (
            cleaned_complaint.lower()
        )

        for aspect, keywords in aspects.items():

            if any(
                keyword in complaint_lower
                for keyword in keywords
            ):

                detected_aspects.append(
                    aspect
                )

        if detected_aspects:

            st.info(
                "🏨 Related Hotel Area: "
                + ", ".join(detected_aspects)
            )

        else:

            st.info(
                "🏨 Related Hotel Area: General Complaint"
            )


# =========================================================
# DEPARTMENT-WISE ACTION BOARD
# =========================================================

st.markdown("---")
st.markdown("## 🏢 Department-wise Action Board")

st.write(
    "This section converts customer complaints into "
    "department-wise actions for hotel management."
)

department_mapping = {
    "Room": "🛏️ Room Management",
    "Cleanliness": "🧹 Housekeeping",
    "Staff": "👥 Front Office & Service",
    "Food": "🍽️ Food & Restaurant",
    "Location": "📍 Management",
    "Wi-Fi": "💻 IT / Technical Support",
    "Facilities": "🔧 Maintenance & Facilities"
}

department_actions = {
    "Room": (
        "Inspect rooms, beds and bathrooms and "
        "resolve maintenance issues."
    ),
    "Cleanliness": (
        "Improve housekeeping checks and room "
        "cleaning procedures."
    ),
    "Staff": (
        "Review service quality and provide staff "
        "training if required."
    ),
    "Food": (
        "Check food quality, temperature, taste "
        "and breakfast service."
    ),
    "Location": (
        "Review guest concerns and highlight "
        "location strengths."
    ),
    "Wi-Fi": (
        "Check internet connectivity and technical "
        "infrastructure."
    ),
    "Facilities": (
        "Inspect facilities such as parking, gym "
        "and swimming pool."
    )
}

department_data = []

for aspect in aspects:

    aspect_mask = pd.Series(
        aspect_masks[aspect],
        index=df.index
    )

    aspect_reviews = df.loc[
        aspect_mask
    ]

    if len(aspect_reviews) > 0:

        negative_reviews = aspect_reviews[
            aspect_reviews["Rating"] <= 2
        ]

        negative_percentage = (
            len(negative_reviews)
            / len(aspect_reviews)
        ) * 100

        if negative_percentage >= 20:
            priority = "🔴 High"

        elif negative_percentage >= 10:
            priority = "🟡 Medium"

        else:
            priority = "🟢 Low"

        department_data.append({
            "Department": department_mapping[
                aspect
            ],
            "Hotel Area": aspect,
            "Negative Reviews": len(
                negative_reviews
            ),
            "Negative %": (
                f"{negative_percentage:.1f}%"
            ),
            "Priority": priority,
            "Recommended Action": (
                department_actions[aspect]
            )
        })


department_df = pd.DataFrame(
    department_data
)

department_df = department_df.sort_values(
    by="Negative Reviews",
    ascending=False
)

st.dataframe(
    department_df,
    use_container_width=True,
    hide_index=True
)

st.markdown("### 🚨 Department Priorities")

high_priority_departments = (
    department_df[
        department_df["Priority"] == "🔴 High"
    ]["Department"].tolist()
)

medium_priority_departments = (
    department_df[
        department_df["Priority"] == "🟡 Medium"
    ]["Department"].tolist()
)

if high_priority_departments:

    st.error(
        "🔴 High Priority: "
        + ", ".join(
            high_priority_departments
        )
    )

if medium_priority_departments:

    st.warning(
        "🟡 Medium Priority: "
        + ", ".join(
            medium_priority_departments
        )
    )

if (
    not high_priority_departments
    and not medium_priority_departments
):

    st.success(
        "🟢 No major department-level problems detected."
    )


# =========================================================
# EARLY WARNING SYSTEM
# =========================================================

st.markdown("---")
st.markdown("## 🚨 Early Warning System")

st.write(
    "This system identifies hotel areas with a high level "
    "of negative customer feedback and alerts the manager."
)

warning_data = []

for aspect in aspects:

    aspect_mask = pd.Series(
        aspect_masks[aspect],
        index=df.index
    )

    aspect_reviews = df.loc[
        aspect_mask
    ]

    if len(aspect_reviews) > 0:

        negative_reviews = aspect_reviews[
            aspect_reviews["Rating"] <= 2
        ]

        negative_percentage = (
            len(negative_reviews)
            / len(aspect_reviews)
        ) * 100

        warning_data.append({
            "Hotel Area": aspect,
            "Total Reviews": len(
                aspect_reviews
            ),
            "Negative Reviews": len(
                negative_reviews
            ),
            "Negative %": negative_percentage
        })


warning_df = pd.DataFrame(
    warning_data
)

warning_df = warning_df.sort_values(
    by="Negative %",
    ascending=False
)

st.markdown("### ⚠️ Risk Analysis")

for _, row in warning_df.iterrows():

    aspect = row["Hotel Area"]
    negative_percentage = row["Negative %"]

    if negative_percentage >= 20:

        st.error(
            f"🚨 HIGH ALERT — {aspect}: "
            f"{negative_percentage:.1f}% negative feedback"
        )

        st.write(
            f"Immediate attention recommended for **{aspect}**."
        )

    elif negative_percentage >= 15:

        st.warning(
            f"⚠️ WARNING — {aspect}: "
            f"{negative_percentage:.1f}% negative feedback"
        )

        st.write(
            f"Monitor **{aspect}** and take corrective action."
        )

    else:

        st.success(
            f"🟢 NORMAL — {aspect}: "
            f"{negative_percentage:.1f}% negative feedback"
        )

st.markdown("### 📊 Complaint Risk by Hotel Area")

st.bar_chart(
    warning_df.set_index("Hotel Area")[
        "Negative %"
    ]
)

st.info(
    "💡 Early warnings are based on negative review percentage "
    "in the current dataset."
)


# =========================================================
# COMPLAINT → ACTION → RESULT LOOP
# =========================================================

st.markdown("---")
st.markdown("## 🔄 Complaint → Action → Result Loop")

st.write(
    "Record the complaint, action taken by the hotel, "
    "and the result after the action."
)

col1, col2 = st.columns(2)

with col1:

    complaint_text = st.text_area(
        "🚨 Complaint",
        placeholder=(
            "Example: Breakfast was cold and tasteless."
        ),
        key="loop_complaint"
    )

    department = st.selectbox(
        "🏢 Department",
        [
            "🧹 Housekeeping",
            "🍽️ Food & Restaurant",
            "👥 Front Office & Service",
            "💻 IT / Technical Support",
            "🔧 Maintenance & Facilities",
            "📍 Management"
        ],
        key="loop_department"
    )

with col2:

    action_taken = st.text_area(
        "🛠️ Action Taken",
        placeholder=(
            "Example: Improved breakfast preparation "
            "and temperature checks."
        ),
        key="loop_action"
    )

    result = st.selectbox(
        "📊 Result After Action",
        [
            "🟢 Improved",
            "🟡 Partially Improved",
            "🔴 Not Improved",
            "⏳ Still Monitoring"
        ],
        key="loop_result"
    )


if st.button(
    "💾 Record Action & Result",
    key="record_action_result"
):

    if (
        complaint_text.strip() == ""
        or action_taken.strip() == ""
    ):

        st.warning(
            "⚠️ Please enter both the complaint and action taken."
        )

    else:

        st.success(
            "✅ Complaint action has been recorded successfully."
        )

        st.markdown("### 📋 Action Summary")

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:

            st.write("🚨 **Complaint**")
            st.info(complaint_text)

            st.write("🏢 **Department**")
            st.info(department)

        with summary_col2:

            st.write("🛠️ **Action Taken**")
            st.info(action_taken)

            st.write("📊 **Result**")
            st.info(result)

        if result == "🟢 Improved":

            st.success(
                "🎉 Great! The problem appears to have improved."
            )

        elif result == "🟡 Partially Improved":

            st.warning(
                "⚠️ The problem has improved partially. "
                "Further action may be required."
            )

        elif result == "🔴 Not Improved":

            st.error(
                "🚨 The problem has not improved. "
                "Immediate corrective action is recommended."
            )

        else:

            st.info(
                "⏳ Continue monitoring this issue."
            )


# =========================================================
# CUSTOMER THANK-YOU CHATBOT
# =========================================================

st.markdown("---")
st.markdown("## 🤖 Customer Thank-You Chatbot")

st.write(
    "Generate a professional hotel response based on customer feedback."
)

chatbot_review = st.text_area(
    "💬 Enter Customer Review",
    placeholder=(
        "Example: The room was clean and "
        "the staff were very friendly."
    ),
    key="chatbot_review_input"
)

if st.button(
    "🤖 Generate Hotel Response",
    key="generate_chatbot_response"
):

    if chatbot_review.strip() == "":

        st.warning(
            "⚠️ Please enter a customer review."
        )

    else:

        cleaned_chatbot_review = clean_text(
            chatbot_review
        )

        chatbot_vector = (
            review_tfidf.transform(
                [cleaned_chatbot_review]
            )
        )

        chatbot_sentiment = (
            review_model.predict(
                chatbot_vector
            )[0]
        )

        chatbot_aspects = []

        for aspect, keywords in aspects.items():

            for keyword in keywords:

                if keyword in cleaned_chatbot_review.lower():

                    chatbot_aspects.append(
                        aspect
                    )

                    break

        # =================================================
        # GENERATE RESPONSE
        # =================================================

        if chatbot_sentiment == "Positive":

            if chatbot_aspects:

                response = (
                    "Thank you for your wonderful feedback! "
                    "We are delighted that you enjoyed our "
                    + ", ".join(chatbot_aspects)
                    + ". Your appreciation means a lot to our team. "
                    "We look forward to welcoming you again!"
                )

            else:

                response = (
                    "Thank you so much for your wonderful feedback! "
                    "We are delighted that you had a pleasant experience "
                    "with us. We look forward to welcoming you again!"
                )

        elif chatbot_sentiment == "Neutral":

            response = (
                "Thank you for sharing your feedback with us. "
                "We truly appreciate your comments. "
                "Your feedback helps us understand our guests' "
                "experiences and improve our services."
            )

        else:

            if chatbot_aspects:

                response = (
                    "We are sorry that your experience did not meet "
                    "your expectations, especially regarding "
                    + ", ".join(chatbot_aspects)
                    + ". Thank you for bringing this to our attention. "
                    "We take your feedback seriously and will work "
                    "to improve our services."
                )

            else:

                response = (
                    "We are sorry that your experience did not meet "
                    "your expectations. Thank you for sharing your "
                    "feedback with us. We take your concerns seriously "
                    "and will work to improve our services."
                )

        # =================================================
        # REVIEW ANALYSIS
        # =================================================

        st.markdown("### 🧠 Review Analysis")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Sentiment",
                chatbot_sentiment
            )

        with col2:

            st.metric(
                "Related Areas",
                len(chatbot_aspects)
            )

        # =================================================
        # RELATED HOTEL AREAS
        # =================================================

        if chatbot_aspects:

            st.info(
                "🏨 Related Hotel Areas: "
                + ", ".join(chatbot_aspects)
            )

        else:

            st.info(
                "🏨 Related Hotel Areas: General Feedback"
            )

        # =================================================
        # SUGGESTED RESPONSE
        # =================================================

        st.markdown("### 💬 Suggested Hotel Response")

        st.success(response)
