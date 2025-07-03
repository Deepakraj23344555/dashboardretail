import streamlit as st
import pandas as pd
import sqlalchemy
from sqlalchemy import text
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import hashlib
from prophet import Prophet
from prophet.plot import plot_components_plotly
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Retail Sales Dashboard", layout="wide")

# -------------------- CUSTOM STYLES --------------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #1e3c72, #2a5298);
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stTabs [data-baseweb="tab"] {
        color: white !important;
    }
    section[data-testid="stSidebar"] .stTabs [aria-selected="true"] {
        font-weight: bold;
        border-bottom: 2px solid #f0b90b;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- DATABASE SETUP --------------------
engine = sqlalchemy.create_engine('sqlite:///sales.db')
user_engine = sqlalchemy.create_engine('sqlite:///users.db')
feedback_engine = sqlalchemy.create_engine('sqlite:///feedback.db')

with user_engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    """))
    conn.commit()

with feedback_engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS feedback (
            username TEXT,
            message TEXT,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """))
    conn.commit()

# -------------------- SESSION STATE --------------------
if 'auth' not in st.session_state:
    st.session_state.auth = False
if 'user' not in st.session_state:
    st.session_state.user = ""
if 'welcome_screen' not in st.session_state:
    st.session_state.welcome_screen = True

# -------------------- WELCOME SCREEN --------------------
if not st.session_state.auth and st.session_state.welcome_screen:
    st.markdown("""
        <style>
        .big-title {
            font-size: 3.5rem;
            font-weight: bold;
            text-align: center;
            background: -webkit-linear-gradient(45deg, #89f7fe, #66a6ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding-top: 2rem;
        }
        .sub-title {
            font-size: 1.5rem;
            text-align: center;
            color: #333;
            margin-bottom: 2rem;
        }
        .cta-button {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-top: 2rem;
        }
        .stButton>button {
            background: linear-gradient(90deg, #66a6ff, #89f7fe);
            color: white;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            border: none;
            font-size: 1.2rem;
            transition: background 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #89f7fe, #66a6ff);
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='big-title'>🚀 Retail Sales Analytics Platform</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Transform Your Sales Data Into Actionable Insights</div>", unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1621523412850-ef3dbd394bb9", use_container_width=True, caption="Data-driven decisions for your business")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔐 Login"):
            st.session_state.welcome_screen = False
    with col2:
        if st.button("📝 Register"):
            st.session_state.welcome_screen = False

    st.stop()

# -------------------- AUTH HELPERS --------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(username, password):
    df = pd.read_sql("SELECT * FROM users WHERE username = ?", user_engine, params=(username,))
    return not df.empty and df['password'][0] == hash_password(password)

def register_user(username, password):
    df = pd.read_sql("SELECT * FROM users WHERE username = ?", user_engine, params=(username,))
    if not df.empty:
        return False
    with user_engine.connect() as conn:
        conn.execute(
            text("INSERT INTO users (username, password) VALUES (:u, :p)"),
            {"u": username, "p": hash_password(password)}
        )
        conn.commit()
    return True

def save_feedback(username, message):
    with feedback_engine.connect() as conn:
        conn.execute(
            text("INSERT INTO feedback (username, message) VALUES (:u, :m)"),
            {"u": username, "m": message}
        )
        conn.commit()

# -------------------- SALES HELPERS --------------------
def clean_sales_data(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    rename_map = {
        "orderdate": "date",
        "order_date": "date",
        "item_type": "product",
        "item": "product",
        "units_sold": "units_sold",
        "total_revenue": "revenue",
        "sales": "revenue"
    }
    for col, new in rename_map.items():
        if col in df.columns:
            df.rename(columns={col: new}, inplace=True)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if 'units_sold' in df.columns:
        df['units_sold'] = pd.to_numeric(df['units_sold'], errors='coerce')
    if 'revenue' in df.columns:
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
    if 'region' not in df.columns:
        df['region'] = 'Unknown'
    return df

def save_to_db(df):
    try:
        df = clean_sales_data(df)
        required = ['date', 'product', 'region', 'units_sold', 'revenue']
        if not all(col in df.columns for col in required):
            st.error(f"❌ Missing required columns: {set(required) - set(df.columns)}")
            return False
        df.dropna(subset=required, inplace=True)
        df.to_sql('sales', engine, if_exists='append', index=False)
        return True
    except Exception as e:
        st.error(f"Error saving to DB: {e}")
        return False

def load_data():
    try:
        df = pd.read_sql("SELECT * FROM sales", engine)
        df = clean_sales_data(df)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def clear_db():
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM sales"))
        conn.commit()

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# -------------------- AUTH UI --------------------
if not st.session_state.auth:
    st.sidebar.title("👤 User Login")
    tab1, tab2 = st.sidebar.tabs(["Login", "Register"])
    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if verify_user(username, password):
                st.session_state.auth = True
                st.session_state.user = username
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials.")
    with tab2:
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        if st.button("Register"):
            if register_user(new_user, new_pass):
                st.success("✅ Registration successful! You can now log in.")
            else:
                st.error("❌ Username already exists.")
    st.stop()

# -------------------- MAIN APP --------------------
st.sidebar.markdown(
    f"<span style='color:white; font-weight:bold;'>👋 Welcome, {st.session_state.user}</span>",
    unsafe_allow_html=True
)
if st.sidebar.button("🚪 Logout"):
    st.session_state.auth = False
    st.session_state.user = ""
    st.session_state.welcome_screen = True
    st.rerun()

menu = ["Upload Data", "View Data", "Dashboard", "Predictions", "Admin Panel", "Feedback"]
choice = st.sidebar.selectbox("📂 Navigate", menu)

# ... keep the rest of your code (Upload, View, Dashboard, Predictions, Admin Panel, Feedback)
# unchanged (you can paste all that logic here from your current app — the *rest* of your original code remains as is)

# -------------------- UPLOAD --------------------
if choice == "Upload Data":
    st.subheader("📤 Upload Sales CSV File")
    with st.expander("📌 CSV Format Example"):
        st.markdown("""
        | order_date  | item_type  | region | units_sold | total_revenue |
        |-------------|------------|--------|------------|---------------|
        | 2024-06-01  | Widget A   | East   | 10         | 100           |
        """)
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        try:
            df = pd.read_csv(file, encoding='latin1')
            st.dataframe(df)
            if st.button("✅ Save to Database"):
                if save_to_db(df):
                    st.success("Saved to database!")
        except Exception as e:
            st.error(f"❌ Error: {e}")
    if st.button("🔄 Clear All Data"):
        clear_db()
        st.success("Sales database cleared.")

# -------------------- VIEW --------------------
elif choice == "View Data":
    st.subheader("📑 View Stored Sales Data")
    data = load_data()
    if data.empty:
        st.warning("⚠ No data found.")
    else:
        st.dataframe(data)
        st.download_button("📥 Download All Data", data=convert_df(data), file_name='sales_data.csv', mime='text/csv')

# -------------------- DASHBOARD --------------------
elif choice == "Dashboard":
    st.subheader("📊 Sales Dashboard")
    data = load_data()
    if data.empty:
        st.warning("⚠ No data found.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            region = st.selectbox("🌍 Select Region", ["All"] + sorted(data['region'].dropna().unique()))
        with col2:
            product = st.selectbox("📦 Select Product", ["All"] + sorted(data['product'].dropna().unique()))
        col3, col4 = st.columns(2)
        with col3:
            start_date = st.date_input("📅 Start Date", data['date'].min())
        with col4:
            end_date = st.date_input("📅 End Date", data['date'].max())
        data = data[(data['date'] >= pd.to_datetime(start_date)) & (data['date'] <= pd.to_datetime(end_date))]
        if region != "All":
            data = data[data['region'] == region]
        if product != "All":
            data = data[data['product'] == product]

        st.markdown("### 📈 Key Performance Indicators")
        c1, c2 = st.columns(2)
        c1.metric("Total Revenue", f"${data['revenue'].sum():,.2f}")
        c2.metric("Units Sold", f"{data['units_sold'].sum():,.0f}")

        st.markdown("### 📅 Revenue Over Time")
        daily = data.groupby('date').agg({'revenue': 'sum'}).reset_index()
        st.plotly_chart(px.line(daily, x='date', y='revenue', markers=True), use_container_width=True)
        st.markdown("""
        **Explanation:**  
        This line chart shows how total revenue changes over time. Peaks may indicate high-demand periods (e.g., promotions or holidays). Use it to spot trends or seasonality in sales.
        """)

        st.markdown("### 📦 Top Selling Products")
        top_products = data.groupby('product')['revenue'].sum().sort_values(ascending=False).reset_index()
        st.plotly_chart(px.bar(top_products, x='product', y='revenue', text_auto=True), use_container_width=True)
        st.markdown("""
        **Explanation:**  
        This bar chart displays total revenue by product. Taller bars mean higher sales. It helps identify best-selling products to prioritize or stock more of.
        """)

        st.markdown("### 🌍 Region vs Product Heatmap")
        pivot = data.pivot_table(values='revenue', index='region', columns='product', aggfunc='sum', fill_value=0)
        fig, ax = plt.subplots()
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
        st.pyplot(fig)
        st.markdown("""
        **Explanation:**  
        This heatmap shows revenue distribution across regions and products. Darker cells mean higher sales. It reveals regional preferences or underperforming areas.
        """)

        st.markdown("### 🗓 Monthly Trend")
        data['month'] = data['date'].dt.to_period('M')
        monthly = data.groupby('month')[['revenue', 'units_sold']].sum().reset_index()
        st.bar_chart(monthly.set_index('month'))
        st.markdown("""
        **Explanation:**  
        This monthly trend chart shows aggregated revenue and units sold over time. It's useful for spotting seasonal patterns, growth trends, or cyclical dips.
        """)

        st.markdown("### 📌 Dynamic Chart")
        colx1, colx2 = st.columns(2)
        xcol = colx1.selectbox("X-axis", options=data.select_dtypes(include=['object', 'datetime64']).columns)
        ycol = colx2.selectbox("Y-axis", options=data.select_dtypes(include='number').columns)
        st.plotly_chart(px.bar(data, x=xcol, y=ycol), use_container_width=True)
        st.markdown(f"""
        **Explanation:**  
        This dynamic chart lets you choose any categorical/date column for the X-axis and any numeric column for the Y-axis. It gives you flexibility to explore relationships in your data.
        """)

        st.markdown("### 🔬 Correlation Matrix")
        st.dataframe(data.corr(numeric_only=True).round(2))
        st.markdown("""
        **Explanation:**  
        This correlation matrix shows relationships between numerical variables. Values closer to 1 or -1 indicate strong positive or negative correlations. It helps identify factors influencing revenue.
        """)

# -------------------- FEEDBACK --------------------
elif choice == "Feedback":
    st.subheader("⭐ Rate Your Experience")
    if st.session_state.get("feedback_submitted", False):
        st.info("📝 You have already submitted feedback. Thank you!")
    else:
        if 'star_rating' not in st.session_state:
            st.session_state.star_rating = 0
        st.markdown("### Select Star Rating:")
        stars = st.columns(5)
        for i in range(5):
            if stars[i].button("⭐" if st.session_state.star_rating > i else "☆", key=f"star{i}"):
                st.session_state.star_rating = i + 1
        st.markdown(f"Your Rating: {st.session_state.star_rating} star{'s' if st.session_state.star_rating > 1 else ''}")
        comment = st.text_area("💬 Any comments? (optional)", max_chars=300)
        if st.button("Submit Feedback"):
            if st.session_state.star_rating == 0:
                st.warning("⚠ Please select a star rating before submitting.")
            else:
                save_feedback(
                    st.session_state.user,
                    f"Rating: {st.session_state.star_rating} stars | Comment: {comment.strip() or 'No comment'}"
                )
                st.success("✅ Thanks for your feedback!")
                st.session_state.feedback_submitted = True
                st.session_state.star_rating = 0

# -------------------- ADMIN PANEL --------------------
elif choice == "Admin Panel":
    st.subheader("🛠 Admin Panel")
    if st.session_state.user != "admin":
        st.warning("⛔ You are not authorized to view this page.")
    else:
        feedback_df = pd.read_sql("SELECT * FROM feedback ORDER BY submitted_at DESC", feedback_engine)
        st.markdown("### 🗣 All Feedback")
        if feedback_df.empty:
            st.info("No feedback submitted yet.")
        else:
            st.dataframe(feedback_df)
            st.markdown("### 📊 Feedback Analytics")
            feedback_df['rating'] = feedback_df['message'].str.extract(r'Rating:\s*(\d+)').astype(float)
            avg_rating = feedback_df['rating'].mean()
            st.metric("Average Rating", f"{avg_rating:.2f} ⭐")
            rating_counts = feedback_df['rating'].value_counts().sort_index()
            chart = pd.DataFrame({"Rating": rating_counts.index, "Count": rating_counts.values})
            st.bar_chart(chart.set_index("Rating"))
            with st.expander("💬 View Sample Comments"):
                comments = feedback_df[['username', 'message', 'submitted_at']].copy()
                comments['Comment'] = comments['message'].str.extract(r'Comment:\s*(.*)')
                st.dataframe(comments[['username', 'submitted_at', 'Comment']])
        st.markdown("### 👥 Registered Users")
        users_df = pd.read_sql("SELECT username FROM users", user_engine)
        st.success(f"Total Users Registered: {users_df.shape[0]}")
        st.dataframe(users_df)

# -------------------- PREDICTIONS --------------------
elif choice == "Predictions":
    st.subheader("🔮 Predictive Insights")
    prediction_option = st.selectbox("Select Prediction Type", [
        "Sales Forecast (Time Series)",
        "Revenue Prediction Model",
        "Seasonality Analysis"
    ])

    # -------------------- Sales Forecast --------------------
    if prediction_option == "Sales Forecast (Time Series)":
        st.markdown("### 📈 Sales Forecast (Time Series)")
        data = load_data()
        if data.empty:
            st.warning("⚠ Not enough data to forecast.")
        else:
            df = data.groupby('date').agg({'revenue': 'sum'}).reset_index().rename(columns={"date": "ds", "revenue": "y"})
            model = Prophet()
            model.fit(df)
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            st.markdown("#### 🔮 Forecasted Revenue (Next 30 Days)")
            st.plotly_chart(px.line(forecast, x='ds', y='yhat', labels={'ds': 'Date', 'yhat': 'Predicted Revenue'}), use_container_width=True)
            st.markdown("""
            **Explanation:**  
            This forecast projects future revenue for the next 30 days based on historical trends. It helps with inventory planning, budgeting, and setting sales targets.
            """)

            st.markdown("#### 📉 Forecast Components")
            st.plotly_chart(plot_components_plotly(model, forecast), use_container_width=True)
            st.markdown("""
            **Explanation:**  
            These components show trend, seasonality, and holiday effects in the forecast model. It helps understand what drives sales fluctuations over time.
            """)

    # -------------------- Revenue Prediction --------------------
    elif prediction_option == "Revenue Prediction Model":
        st.markdown("### 💰 Revenue Prediction Model")
        data = load_data()
        if data.empty:
            st.warning("⚠ Not enough data to train a prediction model.")
        else:
            df = data[['product', 'region', 'units_sold', 'revenue']].dropna()
            X = df[['product', 'region', 'units_sold']]
            y = df['revenue']
            preprocessor = ColumnTransformer([
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['product', 'region'])
            ], remainder='passthrough')
            model = make_pipeline(preprocessor, LinearRegression())
            model.fit(X, y)
            st.markdown("#### 🎯 Predict Revenue for New Entry")
            selected_product = st.selectbox("Select Product", sorted(df['product'].unique()))
            selected_region = st.selectbox("Select Region", sorted(df['region'].unique()))
            units_input = st.number_input("Units Sold", min_value=1, value=10)
            input_df = pd.DataFrame([{
                'product': selected_product,
                'region': selected_region,
                'units_sold': units_input
            }])
            predicted_revenue = model.predict(input_df)[0]
            st.success(f"📈 Predicted Revenue: ${predicted_revenue:.2f}")
            st.markdown("""
            **Explanation:**  
            This model predicts expected revenue based on product, region, and units sold. It helps in setting prices, planning sales, and estimating profits.
            """)

    # -------------------- Seasonality Analysis --------------------
    elif prediction_option == "Seasonality Analysis":
        st.markdown("### 📆 Seasonality Analysis")
        data = load_data()
        if data.empty:
            st.warning("⚠ No data available for seasonality analysis.")
        else:
            data['month'] = data['date'].dt.strftime('%B')
            data['weekday'] = data['date'].dt.strftime('%A')
            st.markdown("#### 📊 Average Revenue by Month")
            monthly_avg = data.groupby('month')['revenue'].mean().reindex([
                'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'
            ])
            st.bar_chart(monthly_avg)
            st.markdown("""
            **Explanation:**  
            This chart shows average revenue by month. Peaks and dips help identify seasonality—when demand is highest or lowest.
            """)

            st.markdown("#### 📅 Average Revenue by Weekday")
            weekday_avg = data.groupby('weekday')['revenue'].mean().reindex([
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
            ])
            st.bar_chart(weekday_avg)
            st.markdown("""
            **Explanation:**  
            This chart shows average revenue by weekday. It reveals which days are busiest, helping schedule staff or promotions effectively.
            """)








