import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set page title and layout
st.set_page_config(page_title="Iris Data Explorer", layout="wide")

# Title and description
st.title("🌸 Iris Dataset Explorer & Classifier")
st.markdown("""
    This app demonstrates data visualization and machine learning prediction using the **Iris dataset**.
    Explore the data, visualize relationships, and predict flower species!
""")

# Load data
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
    return df, iris.target_names

df, class_names = load_data()

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Visualizations", "Predict Species"])

# Sidebar for common filters
st.sidebar.markdown("---")
st.sidebar.subheader("Filter Data")
selected_species = st.sidebar.multiselect(
    "Select Species",
    options=df['species_name'].unique(),
    default=df['species_name'].unique()
)

# Filter dataframe
filtered_df = df[df['species_name'].isin(selected_species)]

# Page 1: Data Overview
if page == "Data Overview":
    st.header("📊 Dataset Overview")
    st.write(f"**Total Records:** {len(filtered_df)}")
    st.write(f"**Filtered Species:** {', '.join(selected_species)}")

    st.dataframe(filtered_df)

    # Download button
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df_to_csv(filtered_df)
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name='iris_filtered.csv',
        mime='text/csv',
    )

# Page 2: Visualizations
elif page == "Visualizations":
    st.header("📈 Interactive Visualizations")

    # Scatter plot
    x_axis = st.selectbox("X-Axis", df.columns[:-2])
    y_axis = st.selectbox("Y-Axis", df.columns[:-2], index=1)

    fig1 = px.scatter(
        filtered_df,
        x=x_axis,
        y=y_axis,
        color='species_name',
        title=f"{y_axis} vs {x_axis} by Species",
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Histogram
    feature = st.selectbox("Select Feature for Histogram", df.columns[:-2])
    fig2 = px.histogram(
        filtered_df,
        x=feature,
        color='species_name',
        title=f"Distribution of {feature}",
        marginal="box",
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    st.plotly_chart(fig2, use_container_width=True)

# Page 3: Predict Species
elif page == "Predict Species":
    st.header("🤖 Predict Species Using ML")

    st.markdown("""
    Enter the flower measurements below to predict its species using a trained **Random Forest model**.
    """)

    # Train model
    X = df.iloc[:, :-2]  # features
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Model accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Model Accuracy:** {accuracy:.2f}")

    # Input fields
    st.subheader("Enter Flower Measurements")
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 3.8)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

    # Predict
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        species = class_names[prediction]
        st.success(f"🌸 Predicted Species: **{species}**")

        # Show feature importance
        st.subheader("Feature Importance")
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values("Importance", ascending=False)

        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title="Feature Importance")
        st.plotly_chart(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Built with ❤️ using Streamlit")
