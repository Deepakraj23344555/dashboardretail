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
# Set the page configuration for the Streamlit app
st.set_page_config(page_title="Retail Sales Dashboard", layout="wide")

# Custom CSS for animations and styling
st.markdown("""
    <style>
        /* Fade-in-up animation for the welcome banner */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .welcome-banner {
            animation: fadeInUp 1s ease-out;
        }
        /* Animated gradient background for the main app */
        .stApp {
            background: linear-gradient(to right, #c4fda1, #c2e9fb, #cfa1fd);
            animation: gradient 15s ease infinite;
            background-size: 400% 400%;
        }
        @keyframes gradient {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        /* Sidebar styling */
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

# Welcome banner at the top of the page
st.markdown("""
    <div class="welcome-banner" style="text-align:center; padding: 2rem 1rem;
            border-radius: 15px; background: linear-gradient(to right, #89f7fe, #66a6ff);
            color: #ffffff; font-size: 2.5rem; font-weight: bold;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            box-shadow: 0 0 20px rgba(0,0,0,0.3);">
        🚀 Welcome to the <span style="color: #ffdf00;">✨ SalesPulse</span>!
    </div>
""", unsafe_allow_html=True)


# -------------------- DATABASES --------------------
# Initialize SQLite database engines
engine = sqlalchemy.create_engine('sqlite:///sales.db')
user_engine = sqlalchemy.create_engine('sqlite:///users.db')
feedback_engine = sqlalchemy.create_engine('sqlite:///feedback.db')

# Create 'users' table if it doesn't exist
with user_engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    """))
    conn.commit()

# Create 'feedback' table if it doesn't exist
with feedback_engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS feedback (
            username TEXT,
            message TEXT,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """))
    conn.commit()


# -------------------- AUTH HELPERS --------------------
def hash_password(password):
    """Hashes a password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(username, password):
    """Verifies user credentials against the database."""
    df = pd.read_sql("SELECT * FROM users WHERE username = ?", user_engine, params=(username,))
    return not df.empty and df['password'][0] == hash_password(password)

def register_user(username, password):
    """Registers a new user if the username doesn't already exist."""
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
    """Saves user feedback to the database."""
    with feedback_engine.connect() as conn:
        conn.execute(
            text("INSERT INTO feedback (username, message) VALUES (:u, :m)"),
            {"u": username, "m": message}
        )
        conn.commit()


# -------------------- SALES HELPERS --------------------
def clean_sales_data(df):
    """Cleans and standardizes the sales DataFrame."""
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    rename_map = {
        "orderdate": "date", "order_date": "date",
        "item_type": "product", "item": "product",
        "units_sold": "units_sold",
        "total_revenue": "revenue", "sales": "revenue"
    }
    df.rename(columns={col: new for col, new in rename_map.items() if col in df.columns}, inplace=True)
    
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
    """Saves a DataFrame to the sales database after cleaning."""
    try:
        df = clean_sales_data(df)
        required = ['date', 'product', 'region', 'units_sold', 'revenue']
        if not all(col in df.columns for col in required):
            st.error(f"✖️ Missing required columns: {set(required) - set(df.columns)}")
            return False
        df.dropna(subset=required, inplace=True)
        df.to_sql('sales', engine, if_exists='append', index=False)
        return True
    except Exception as e:
        st.error(f"Error saving to DB: {e}")
        return False

def load_data():
    """Loads sales data from the database."""
    try:
        df = pd.read_sql("SELECT * FROM sales", engine)
        return clean_sales_data(df)
    except Exception as e:
        # If the table doesn't exist yet, return an empty DataFrame
        if "no such table" in str(e):
            return pd.DataFrame()
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def clear_db():
    """Deletes all records from the sales table."""
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM sales"))
        conn.commit()

@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a UTF-8 encoded CSV file for downloading."""
    return df.to_csv(index=False).encode('utf-8')


# -------------------- SESSION SETUP --------------------
if 'auth' not in st.session_state:
    st.session_state.auth = False
if 'user' not in st.session_state:
    st.session_state.user = ""


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
                st.success("✔️ Login successful!")
                st.rerun()
            else:
                st.error("✖️ Invalid credentials.")
    with tab2:
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        if st.button("Register"):
            if register_user(new_user, new_pass):
                st.success("✔️ Registration successful! You can now log in.")
            else:
                st.error("✖️ Username already exists.")
    st.stop()


# -------------------- APP HEADER & LOGOUT --------------------
st.sidebar.markdown(
    f"<span style='color:white; font-weight:bold;'>🚀 Welcome, {st.session_state.user}</span>",
    unsafe_allow_html=True
)
if st.sidebar.button("🚪 Logout"):
    st.session_state.auth = False
    st.session_state.user = ""
    st.rerun()


# -------------------- MAIN MENU --------------------
menu = ["Upload Data", "View Data", "Dashboard", "Predictions", "Feedback", "Admin Panel"]
choice = st.sidebar.selectbox("🗂️ Navigate", menu)


# -------------------- UPLOAD PAGE --------------------
if choice == "Upload Data":
    st.subheader("📤 Upload Sales CSV File")
    with st.expander("📍 CSV Format Example"):
        st.markdown("""
        Your CSV should have columns like these. The names can vary slightly.
        | order_date  | item_type  | region | units_sold | total_revenue |
        |-------------|------------|--------|------------|---------------|
        | 2024-06-01  | Widget A   | East   | 10         | 100           |
        """)
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        try:
            df = pd.read_csv(file, encoding='latin1')
            st.dataframe(df.head())
            if st.button("✔️ Save to Database"):
                if save_to_db(df):
                    st.success("Saved to database!")
        except Exception as e:
            st.error(f"✖️ Error processing file: {e}")
    if st.button("🔄 Clear All Sales Data"):
        clear_db()
        st.success("Sales database has been cleared.")


# -------------------- VIEW PAGE --------------------
elif choice == "View Data":
    st.subheader("📑 View Stored Sales Data")
    data = load_data()
    if data.empty:
        st.warning("⚠️ No data found. Please upload a file on the 'Upload Data' page.")
    else:
        st.dataframe(data)
        st.download_button(
            "📥 Download All Data as CSV", 
            data=convert_df_to_csv(data), 
            file_name='sales_data.csv', 
            mime='text/csv'
        )


# -------------------- DASHBOARD PAGE --------------------
elif choice == "Dashboard":
    st.subheader("📊 Sales Dashboard")
    data = load_data()
    if data.empty:
        st.warning("⚠️ No data found. Please upload a file to view the dashboard.")
    else:
        # --- FILTERS ---
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
        
        # Apply filters
        filtered_data = data[
            (data['date'] >= pd.to_datetime(start_date)) & 
            (data['date'] <= pd.to_datetime(end_date))
        ]
        if region != "All":
            filtered_data = filtered_data[filtered_data['region'] == region]
        if product != "All":
            filtered_data = filtered_data[filtered_data['product'] == product]

        if filtered_data.empty:
            st.warning("No data matches the selected filters.")
        else:
            # --- KPIS ---
            st.markdown("### 📈 Key Performance Indicators")
            c1, c2 = st.columns(2)
            c1.metric("Total Revenue", f"${filtered_data['revenue'].sum():,.2f}")
            c2.metric("Units Sold", f"{filtered_data['units_sold'].sum():,.0f}")

            # --- CHARTS ---
            st.markdown("### 📅 Revenue Over Time")
            daily = filtered_data.groupby('date').agg({'revenue': 'sum'}).reset_index()
            st.plotly_chart(px.line(daily, x='date', y='revenue', markers=True), use_container_width=True)

            st.markdown("### 📦 Top Selling Products by Revenue")
            top_products = filtered_data.groupby('product')['revenue'].sum().nlargest(10).sort_values(ascending=False).reset_index()
            st.plotly_chart(px.bar(top_products, x='product', y='revenue', text_auto='.2s'), use_container_width=True)

            st.markdown("### 🌍 Region vs Product Revenue Heatmap")
            pivot = filtered_data.pivot_table(values='revenue', index='region', columns='product', aggfunc='sum', fill_value=0)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
            st.pyplot(fig)

            st.download_button(
                "📥 Download Filtered Data", 
                data=convert_df_to_csv(filtered_data), 
                file_name="filtered_sales.csv", 
                mime='text/csv'
            )


# -------------------- PREDICTIONS PAGE --------------------
elif choice == "Predictions":
    st.subheader("🔮 Predictive Insights")
    data = load_data()
    if data.empty or len(data) < 10:
        st.warning("⚠️ Not enough data to generate predictions. Please upload more data.")
    else:
        prediction_option = st.selectbox("Select Prediction Type", [
            "Sales Forecast (Time Series)",
            "Revenue Prediction Model",
            "Seasonality Analysis"
        ])

        if prediction_option == "Sales Forecast (Time Series)":
            st.markdown("### 📈 Sales Forecast (Time Series)")
            df_prophet = data.groupby('date').agg({'revenue': 'sum'}).reset_index().rename(columns={"date": "ds", "revenue": "y"})
            if len(df_prophet) < 2:
                st.warning("Need at least 2 data points for forecasting.")
            else:
                model = Prophet()
                model.fit(df_prophet)
                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)
                st.markdown("#### 🔮 Forecasted Revenue (Next 30 Days)")
                fig_forecast = px.line(forecast, x='ds', y='yhat', labels={'ds': 'Date', 'yhat': 'Predicted Revenue'}, title="Forecast")
                fig_forecast.add_scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Actual Revenue')
                st.plotly_chart(fig_forecast, use_container_width=True)
                st.markdown("#### 📉 Forecast Components")
                st.plotly_chart(plot_components_plotly(model, forecast), use_container_width=True)

        elif prediction_option == "Revenue Prediction Model":
            st.markdown("### 💰 Revenue Prediction Model")
            df_model = data[['product', 'region', 'units_sold', 'revenue']].dropna()
            X = df_model[['product', 'region', 'units_sold']]
            y = df_model['revenue']
            
            preprocessor = ColumnTransformer(
                [('cat', OneHotEncoder(handle_unknown='ignore'), ['product', 'region'])], 
                remainder='passthrough'
            )
            model = make_pipeline(preprocessor, LinearRegression())
            model.fit(X, y)
            
            st.markdown("#### 🎯 Predict Revenue for New Entry")
            selected_product = st.selectbox("Select Product", sorted(df_model['product'].unique()))
            selected_region = st.selectbox("Select Region", sorted(df_model['region'].unique()))
            units_input = st.number_input("Units Sold", min_value=1, value=10)
            
            input_df = pd.DataFrame({
                'product': [selected_product],
                'region': [selected_region],
                'units_sold': [units_input]
            })
            predicted_revenue = model.predict(input_df)[0]
            st.success(f"🚀 Predicted Revenue: ${predicted_revenue:,.2f}")

        elif prediction_option == "Seasonality Analysis":
            st.markdown("### 📆 Seasonality Analysis")
            data['month'] = data['date'].dt.strftime('%B')
            data['weekday'] = data['date'].dt.strftime('%A')
            
            st.markdown("#### 📊 Average Revenue by Month")
            monthly_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
            monthly_avg = data.groupby('month')['revenue'].mean().reindex(monthly_order).dropna()
            st.bar_chart(monthly_avg)

            st.markdown("#### 📅 Average Revenue by Weekday")
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekday_avg = data.groupby('weekday')['revenue'].mean().reindex(weekday_order).dropna()
            st.bar_chart(weekday_avg)


# -------------------- FEEDBACK PAGE --------------------
elif choice == "Feedback":
    st.subheader("🌟 Rate Your Experience")
    if st.session_state.get("feedback_submitted", False):
        st.info("📝 You have already submitted feedback. Thank you!")
    else:
        if 'star_rating' not in st.session_state:
            st.session_state.star_rating = 0
            
        st.markdown("### Select Star Rating:")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            if col.button("🌟" if st.session_state.star_rating > i else "☆", key=f"star{i}"):
                st.session_state.star_rating = i + 1
        
        st.markdown(f"Your Rating: {st.session_state.star_rating} star{'s' if st.session_state.star_rating > 1 else ''}")
        comment = st.text_area("💬 Any comments? (optional)", max_chars=300)
        
        if st.button("Submit Feedback"):
            if st.session_state.star_rating == 0:
                st.warning("⚠️ Please select a star rating before submitting.")
            else:
                feedback_message = f"Rating: {st.session_state.star_rating} stars | Comment: {comment.strip() or 'No comment'}"
                save_feedback(st.session_state.user, feedback_message)
                st.success("✔️ Thanks for your feedback!")
                st.session_state.feedback_submitted = True
                st.session_state.star_rating = 0 # Reset for next session
                st.rerun()


# -------------------- ADMIN PANEL PAGE --------------------
elif choice == "Admin Panel":
    st.subheader("Admin Panel")
    if st.session_state.user != "admin":
        st.warning("⛔ You are not authorized to view this page.")
    else:
        admin_tab1, admin_tab2 = st.tabs(["Feedback", "Users"])
        
        with admin_tab1:
            st.markdown("### All Feedback")
            try:
                feedback_df = pd.read_sql("SELECT * FROM feedback ORDER BY submitted_at DESC", feedback_engine)
                if feedback_df.empty:
                    st.info("No feedback submitted yet.")
                else:
                    feedback_df['rating'] = feedback_df['message'].str.extract(r'Rating:\s*(\d+)').astype(float)
                    avg_rating = feedback_df['rating'].mean()
                    
                    st.metric("Average Rating", f"{avg_rating:.2f} 🌟")
                    
                    rating_counts = feedback_df['rating'].value_counts().sort_index()
                    st.bar_chart(rating_counts)
                    
                    with st.expander("View All Feedback Entries"):
                        st.dataframe(feedback_df)
            except Exception as e:
                st.error(f"Could not load feedback: {e}")

        with admin_tab2:
            st.markdown("### Registered Users")
            try:
                users_df = pd.read_sql("SELECT username FROM users", user_engine)
                st.metric("Total Users Registered", f"{users_df.shape[0]}")
                st.dataframe(users_df)
            except Exception as e:
                st.error(f"Could not load users: {e}")
