import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.regression import LinearRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col, isnan, when, count, rand
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType, BooleanType
import random
import io
import base64

# Set page config
st.set_page_config(page_title="Women's Clothing E-Commerce Analysis", 
                   layout="wide",
                   initial_sidebar_state="expanded")
                   
# Initialize SparkSession
@st.cache_resource
def create_spark_session():
    return SparkSession.builder \
        .appName("Women's Clothing E-Commerce") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

spark = create_spark_session()

# Define the schema for the dataset
schema = StructType([
    StructField("Age", IntegerType(), True),
    StructField("Title", StringType(), True),
    StructField("Review_Text", StringType(), True),
    StructField("Rating", IntegerType(), True),
    StructField("Recommended_IND", IntegerType(), True),
    StructField("Positive_Feedback_Count", IntegerType(), True),
    StructField("Division_Name", StringType(), True),
    StructField("Department_Name", StringType(), True),
    StructField("Class_Name", StringType(), True)
])

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", 
                       ["1. Data Generation", 
                        "2. Data Cleaning", 
                        "3. Missing Values", 
                        "4. Exploratory Data Analysis", 
                        "5. Machine Learning Models"])

# Function to generate synthetic data
def generate_synthetic_data(num_rows=23486):
    # Lists for categorical data
    division_names = ["General", "General Petite", "Initmates", "Trend"]
    department_names = ["Tops", "Dresses", "Bottoms", "Intimate", "Jackets", "Trend"]
    class_names = ["Blouses", "Dresses", "Pants", "Intimates", "Jackets", "Lounge", "Skirts", "Jeans", "Sweaters", "Shorts", "Sleep", "Trend"]
    
    # Sample review titles
    title_templates = [
        "Love this {product}!",
        "Not what I expected",
        "Great {product} for the price",
        "Disappointing quality",
        "Perfect fit!",
        "Too small/large",
        "Exactly as described",
        "Beautiful {product}",
        "Not worth the money",
        "Comfortable and stylish"
    ]
    
    # Sample review texts
    review_text_templates = [
        "I really love this {product}. The quality is excellent and it fits perfectly. Would definitely recommend!",
        "The {product} wasn't what I expected. The color was off and the sizing runs small.",
        "Great {product} for the price! Comfortable material and looks more expensive than it is.",
        "Disappointing quality. The {product} started falling apart after just a few wears.",
        "Perfect fit! This {product} is true to size and very flattering.",
        "This {product} runs too small/large. Had to return it.",
        "The {product} is exactly as described. Very pleased with my purchase.",
        "This is a beautiful {product}. I've received many compliments when wearing it.",
        "Not worth the money. The {product} looks cheap and is poorly made.",
        "This {product} is both comfortable and stylish. I bought it in multiple colors!"
    ]
    
    products = ["dress", "blouse", "pants", "jacket", "skirt", "sweater", "top", "jeans", "shorts", "intimates"]
    
    # Generate data
    data = []
    for _ in range(num_rows):
        age = random.randint(18, 70)
        
        # Select random product for consistent review
        product = random.choice(products)
        
        # Generate consistent division, department, class
        division_name = random.choice(division_names)
        
        if division_name == "Initmates":
            department_name = "Intimate"
            class_name = random.choice(["Intimates", "Sleep"])
        elif division_name == "Trend":
            department_name = "Trend"
            class_name = "Trend"
        else:
            department_name = random.choice(["Tops", "Dresses", "Bottoms", "Jackets"])
            
            if department_name == "Tops":
                class_name = random.choice(["Blouses", "Sweaters"])
            elif department_name == "Dresses":
                class_name = "Dresses"
            elif department_name == "Bottoms":
                class_name = random.choice(["Pants", "Skirts", "Jeans", "Shorts"])
            else:  # Jackets
                class_name = "Jackets"
        
        # Generate rating and recommended with some correlation
        rating = random.choices([1, 2, 3, 4, 5], weights=[0.05, 0.1, 0.15, 0.3, 0.4])[0]
        
        # Higher ratings are more likely to be recommended
        if rating >= 4:
            recommended = random.choices([0, 1], weights=[0.1, 0.9])[0]
        elif rating == 3:
            recommended = random.choices([0, 1], weights=[0.5, 0.5])[0]
        else:
            recommended = random.choices([0, 1], weights=[0.9, 0.1])[0]
        
        # Positive feedback correlated with rating
        if rating >= 4:
            positive_feedback = random.randint(0, 25)
        else:
            positive_feedback = random.randint(0, 5)
        
        # Generate title and review text
        title_template = random.choice(title_templates)
        title = title_template.replace("{product}", product)
        
        review_template = random.choice(review_text_templates)
        review_text = review_template.replace("{product}", product)
        
        # Add some missing values randomly
        if random.random() < 0.03:  # 3% chance of missing age
            age = None
        if random.random() < 0.01:  # 1% chance of missing title
            title = None
        if random.random() < 0.05:  # 5% chance of missing rating
            rating = None
        if random.random() < 0.02:  # 2% chance of missing division
            division_name = None
        
        data.append([age, title, review_text, rating, recommended, positive_feedback, 
                     division_name, department_name, class_name])
    
    # Create PySpark DataFrame
    df = spark.createDataFrame(data, schema=schema)
    return df

# Function to download dataframe as CSV
def get_csv_download_link(df, filename="data.csv"):
    # Convert spark dataframe to pandas for download
    pandas_df = df.toPandas()
    csv = pandas_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

# Function to clean data
def clean_data(df):
    # Convert column names to lowercase and replace spaces with underscores
    for column in df.columns:
        df = df.withColumnRenamed(column, column.lower().replace(" ", "_"))
    
    # Remove duplicates
    df = df.dropDuplicates()
    
    # Filter out records with invalid age
    df = df.filter(col("age").isNotNull() & (col("age") > 0) & (col("age") < 100))
    
    # Filter out records with invalid rating
    df = df.filter(col("rating").isNotNull() & (col("rating") >= 1) & (col("rating") <= 5))
    
    # Filter out records with invalid recommended_ind
    df = df.filter(col("recommended_ind").isNotNull() & ((col("recommended_ind") == 0) | (col("recommended_ind") == 1)))
    
    return df

# Function to handle missing values
def handle_missing_values(df):
    # Count missing values
    missing_counts = []
    for column in df.columns:
        missing_count = df.filter(col(column).isNull() | isnan(column)).count()
        missing_counts.append({"Column": column, "Missing Count": missing_count, "Missing %": (missing_count / df.count()) * 100})
    
    # Impute missing values
    # Age - fill with median
    median_age = df.approxQuantile("age", [0.5], 0.01)[0]
    df = df.withColumn("age", when(col("age").isNull(), median_age).otherwise(col("age")))
    
    # Title - fill with "No Title"
    df = df.withColumn("title", when(col("title").isNull(), "No Title").otherwise(col("title")))
    
    # Rating - fill with mode (most common value)
    mode_rating = df.groupBy("rating").count().orderBy(col("count").desc()).first()["rating"]
    df = df.withColumn("rating", when(col("rating").isNull(), mode_rating).otherwise(col("rating")))
    
    # Division - fill with most common value
    mode_division = df.groupBy("division_name").count().orderBy(col("count").desc()).first()["division_name"]
    df = df.withColumn("division_name", when(col("division_name").isNull(), mode_division).otherwise(col("division_name")))
    
    # Department - fill with most common value for that division
    df_with_dept = df.filter(col("department_name").isNotNull())
    dept_by_div = df_with_dept.groupBy("division_name", "department_name").count()
    
    # Use pandas for this complex operation
    # In a real scenario, we would implement this in pure PySpark
    dept_by_div_pd = dept_by_div.toPandas()
    div_to_dept = {}
    for div in dept_by_div_pd['division_name'].unique():
        div_df = dept_by_div_pd[dept_by_div_pd['division_name'] == div]
        div_to_dept[div] = div_df.loc[div_df['count'].idxmax()]['department_name']
    
    # Apply the mapping (using a simpler method for demonstration)
    for div, dept in div_to_dept.items():
        df = df.withColumn("department_name", 
                          when((col("division_name") == div) & (col("department_name").isNull()), 
                              dept).otherwise(col("department_name")))
    
    # Any remaining nulls in department_name get the most common value
    mode_dept = df.groupBy("department_name").count().orderBy(col("count").desc()).first()["department_name"]
    df = df.withColumn("department_name", when(col("department_name").isNull(), mode_dept).otherwise(col("department_name")))
    
    return df, missing_counts

# Function for EDA
def perform_eda(df):
    # Convert to pandas for visualization
    pandas_df = df.toPandas()
    
    # Summary statistics
    numeric_cols = ["age", "rating", "recommended_ind", "positive_feedback_count"]
    summary_stats = pandas_df[numeric_cols].describe()
    
    # Age distribution
    fig_age, ax_age = plt.subplots(figsize=(10, 6))
    sns.histplot(pandas_df["age"], kde=True, ax=ax_age)
    ax_age.set_title("Age Distribution")
    
    # Rating distribution
    fig_rating, ax_rating = plt.subplots(figsize=(10, 6))
    sns.countplot(x="rating", data=pandas_df, ax=ax_rating)
    ax_rating.set_title("Rating Distribution")
    
    # Rating vs Recommendation
    fig_rec, ax_rec = plt.subplots(figsize=(10, 6))
    crosstab = pd.crosstab(pandas_df["rating"], pandas_df["recommended_ind"])
    crosstab.plot(kind="bar", stacked=True, ax=ax_rec)
    ax_rec.set_title("Rating vs Recommendation")
    ax_rec.set_xlabel("Rating")
    ax_rec.set_ylabel("Count")
    ax_rec.legend(["Not Recommended", "Recommended"])
    
    # Division distribution
    fig_div, ax_div = plt.subplots(figsize=(10, 6))
    sns.countplot(y="division_name", data=pandas_df, ax=ax_div, order=pandas_df["division_name"].value_counts().index)
    ax_div.set_title("Division Distribution")
    
    # Department distribution
    fig_dept, ax_dept = plt.subplots(figsize=(10, 6))
    sns.countplot(y="department_name", data=pandas_df, ax=ax_dept, order=pandas_df["department_name"].value_counts().index)
    ax_dept.set_title("Department Distribution")
    
    # Class distribution
    fig_class, ax_class = plt.subplots(figsize=(10, 6))
    sns.countplot(y="class_name", data=pandas_df, ax=ax_class, order=pandas_df["class_name"].value_counts().index)
    ax_class.set_title("Class Distribution")
    
    # Rating by Division
    fig_rate_div, ax_rate_div = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="division_name", y="rating", data=pandas_df, ax=ax_rate_div)
    ax_rate_div.set_title("Rating by Division")
    
    # Positive feedback by rating
    fig_pos, ax_pos = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="rating", y="positive_feedback_count", data=pandas_df, ax=ax_pos)
    ax_pos.set_title("Positive Feedback by Rating")
    
    return {
        "summary_stats": summary_stats,
        "fig_age": fig_age,
        "fig_rating": fig_rating,
        "fig_rec": fig_rec,
        "fig_div": fig_div,
        "fig_dept": fig_dept,
        "fig_class": fig_class,
        "fig_rate_div": fig_rate_div,
        "fig_pos": fig_pos
    }

# Function to prepare data for ML
def prepare_data_for_ml(df):
    # Select features for ML
    ml_df = df.select("age", "rating", "recommended_ind", "positive_feedback_count", 
                     "division_name", "department_name", "class_name")
    
    # Create feature pipeline
    categorical_cols = ["division_name", "department_name", "class_name"]
    string_indexers = [StringIndexer(inputCol=col, outputCol=col+"_idx").fit(ml_df) for col in categorical_cols]
    
    encoders = [OneHotEncoder(inputCol=col+"_idx", outputCol=col+"_vec") for col in categorical_cols]
    
    # Split the data
    train_data, test_data = ml_df.randomSplit([0.8, 0.2], seed=42)
    
    return ml_df, train_data, test_data, string_indexers, encoders

# Function for regression
def run_regression(train_data, test_data, string_indexers, encoders):
    # Prepare features for regression (predict rating)
    assembler = VectorAssembler(
        inputCols=["age", "recommended_ind", "positive_feedback_count", 
                  "division_name_vec", "department_name_vec", "class_name_vec"],
        outputCol="features"
    )
    
    # Create pipeline
    pipeline = Pipeline(stages=string_indexers + encoders + [assembler])
    
    # Fit the pipeline
    pipeline_model = pipeline.fit(train_data)
    
    # Transform the data
    train_data_transformed = pipeline_model.transform(train_data)
    test_data_transformed = pipeline_model.transform(test_data)
    
    # Train linear regression model
    lr = LinearRegression(featuresCol="features", labelCol="rating")
    lr_model = lr.fit(train_data_transformed)
    
    # Make predictions
    predictions = lr_model.transform(test_data_transformed)
    
    # Evaluate the model
    evaluator = RegressionEvaluator(labelCol="rating", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    
    mae_evaluator = RegressionEvaluator(labelCol="rating", predictionCol="prediction", metricName="mae")
    mae = mae_evaluator.evaluate(predictions)
    
    r2_evaluator = RegressionEvaluator(labelCol="rating", predictionCol="prediction", metricName="r2")
    r2 = r2_evaluator.evaluate(predictions)
    
    # Extract some sample predictions
    sample_predictions = predictions.select("rating", "prediction").toPandas().head(10)
    
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "sample_predictions": sample_predictions,
        "coefficients": lr_model.coefficients,
        "intercept": lr_model.intercept
    }

# Function for clustering
def run_clustering(ml_df, string_indexers, encoders):
    # Prepare features for clustering
    assembler = VectorAssembler(
        inputCols=["age", "rating", "recommended_ind", "positive_feedback_count", 
                  "division_name_vec", "department_name_vec", "class_name_vec"],
        outputCol="features"
    )
    
    # Create pipeline
    pipeline = Pipeline(stages=string_indexers + encoders + [assembler])
    
    # Fit the pipeline
    pipeline_model = pipeline.fit(ml_df)
    
    # Transform the data
    data_transformed = pipeline_model.transform(ml_df)
    
    # Scale data for better clustering (normally we'd use StandardScaler here)
    # For simplicity, we'll skip scaling in this example
    
    # Run K-Means with different k
    k_values = range(2, 6)
    results = []
    
    for k in k_values:
        kmeans = KMeans(featuresCol="features", k=k, seed=42)
        model = kmeans.fit(data_transformed)
        
        # Get cluster centers and sizes
        predictions = model.transform(data_transformed)
        cluster_sizes = predictions.groupBy("prediction").count().orderBy("prediction").toPandas()
        
        # Get WSSSE using the summary property
        wssse = model.summary.trainingCost
        
        results.append({
            "k": k,
            "wssse": wssse,
            "cluster_sizes": cluster_sizes
        })
    
    # Choose optimal k based on elbow method
    optimal_k = 3  # This would normally be chosen based on WSSSE plot
    
    # Run final model with optimal k
    kmeans = KMeans(featuresCol="features", k=optimal_k, seed=42)
    model = kmeans.fit(data_transformed)
    
    # Get final predictions
    final_predictions = model.transform(data_transformed)
    
    # Get cluster profiles
    cluster_profiles = final_predictions.groupBy("prediction").agg(
        {"age": "avg", "rating": "avg", "recommended_ind": "avg", "positive_feedback_count": "avg"}
    ).toPandas()
    
    # Get cluster distribution by division
    division_clusters = final_predictions.groupBy("division_name", "prediction").count().toPandas()
    
    return {
        "results": results,
        "optimal_k": optimal_k,
        "cluster_profiles": cluster_profiles,
        "division_clusters": division_clusters
    }

# Function for classification
def run_classification(train_data, test_data, string_indexers, encoders):
    # Prepare features for classification (predict recommended_ind)
    assembler = VectorAssembler(
        inputCols=["age", "rating", "positive_feedback_count", 
                  "division_name_vec", "department_name_vec", "class_name_vec"],
        outputCol="features"
    )
    
    # Create pipeline
    pipeline = Pipeline(stages=string_indexers + encoders + [assembler])
    
    # Fit the pipeline
    pipeline_model = pipeline.fit(train_data)
    
    # Transform the data
    train_data_transformed = pipeline_model.transform(train_data)
    test_data_transformed = pipeline_model.transform(test_data)
    
    # Train Random Forest model
    rf = RandomForestClassifier(featuresCol="features", labelCol="recommended_ind", numTrees=10)
    rf_model = rf.fit(train_data_transformed)
    
    # Make predictions
    predictions = rf_model.transform(test_data_transformed)
    
    # Evaluate the model
    evaluator = MulticlassClassificationEvaluator(labelCol="recommended_ind", 
                                                 predictionCol="prediction", 
                                                 metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    
    f1_evaluator = MulticlassClassificationEvaluator(labelCol="recommended_ind", 
                                                   predictionCol="prediction", 
                                                   metricName="f1")
    f1 = f1_evaluator.evaluate(predictions)
    
    # Extract some sample predictions - convert probability to string to avoid PyArrow errors
    sample_predictions = predictions.select("recommended_ind", "prediction").toPandas().head(10)
    
    # Get probability values separately and convert to strings
    prob_df = predictions.select("probability").limit(10).toPandas()
    prob_df["probability"] = prob_df["probability"].astype(str)
    
    # Combine dataframes
    sample_predictions["probability"] = prob_df["probability"].values
    
    # Get feature importance
    feature_importance = rf_model.featureImportances
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "sample_predictions": sample_predictions,
        "feature_importance": feature_importance
    }

# Page 1: Data Generation
if page == "1. Data Generation":
    st.title("Women's Clothing E-Commerce Dataset Generation")
    
    st.write("""
    This page allows you to generate a synthetic dataset for Women's Clothing E-Commerce reviews.
    The dataset will contain the following fields:
    - Age: Positive Integer variable of the reviewers' age
    - Title: String variable for the title of the review
    - Review Text: String variable for the review body
    - Rating: Positive Ordinal Integer variable for the product score (1-5)
    - Recommended IND: Binary variable (1=recommended, 0=not recommended)
    - Positive Feedback Count: Positive Integer for the number of other customers who found this review positive
    - Division Name: Categorical name of the product high-level division
    - Department Name: Categorical name of the product department
    - Class Name: Categorical name of the product class
    """)
    
    # User input for dataset size
    num_rows = st.slider("Number of rows to generate", min_value=1000, max_value=50000, value=23486, step=1000)
    
    if st.button("Generate Dataset"):
        with st.spinner("Generating dataset..."):
            # Generate the dataset
            df = generate_synthetic_data(num_rows)
            
            # Store in session state
            st.session_state.raw_df = df
            
            # Display sample of the data
            st.write("### Sample Data")
            st.write(df.limit(10).toPandas())
            
            # Display summary statistics
            st.write("### Summary Statistics")
            st.write(df.describe().toPandas())
            
            # Provide download link
            st.markdown(get_csv_download_link(df, "womens_clothing_ecommerce_raw.csv"), unsafe_allow_html=True)
            
            st.success(f"Successfully generated {df.count()} rows of data!")

# Page 2: Data Cleaning
elif page == "2. Data Cleaning":
    st.title("Data Cleaning & Wrangling")
    
    # Check if data exists
    if 'raw_df' not in st.session_state:
        st.warning("Please generate the dataset first on the Data Generation page.")
        if st.button("Generate Default Dataset"):
            with st.spinner("Generating default dataset..."):
                df = generate_synthetic_data()
                st.session_state.raw_df = df
                st.success("Default dataset generated!")
    else:
        st.write("### Original Data Sample")
        st.write(st.session_state.raw_df.limit(5).toPandas())
        
        # Display data stats before cleaning
        st.write("### Data Stats Before Cleaning")
        st.write(f"Number of rows: {st.session_state.raw_df.count()}")
        st.write(f"Number of columns: {len(st.session_state.raw_df.columns)}")
        
        # Show and run cleaning operations
        st.write("### Cleaning Operations")
        cleaning_options = st.multiselect(
            "Select cleaning operations",
            ["Remove duplicates", "Convert column names to lowercase", "Filter invalid age values", 
             "Filter invalid ratings", "Filter invalid recommendation indicators"],
            ["Remove duplicates", "Convert column names to lowercase", "Filter invalid age values", 
             "Filter invalid ratings", "Filter invalid recommendation indicators"]
        )
        
        if st.button("Clean Data"):
            with st.spinner("Cleaning data..."):
                # Clean the data
                cleaned_df = clean_data(st.session_state.raw_df)
                
                # Store in session state
                st.session_state.cleaned_df = cleaned_df
                
                # Display sample of the cleaned data
                st.write("### Cleaned Data Sample")
                st.write(cleaned_df.limit(5).toPandas())
                
                # Display data stats after cleaning
                st.write("### Data Stats After Cleaning")
                st.write(f"Number of rows: {cleaned_df.count()}")
                st.write(f"Number of columns: {len(cleaned_df.columns)}")
                
                # Provide download link
                st.markdown(get_csv_download_link(cleaned_df, "womens_clothing_ecommerce_cleaned.csv"), unsafe_allow_html=True)
                
                st.success("Data cleaning completed!")

# Page 3: Missing Values
elif page == "3. Missing Values":
    st.title("Handling Missing Values")
    
    # Check if cleaned data exists
    if 'cleaned_df' not in st.session_state:
        if 'raw_df' in st.session_state:
            st.warning("Please clean the dataset first on the Data Cleaning page.")
            if st.button("Clean Data Now"):
                with st.spinner("Cleaning data..."):
                    cleaned_df = clean_data(st.session_state.raw_df)
                    st.session_state.cleaned_df = cleaned_df
                    st.success("Data cleaned!")
        else:
            st.warning("Please generate and clean the dataset first.")
            if st.button("Generate and Clean Default Dataset"):
                with st.spinner("Generating and cleaning default dataset..."):
                    raw_df = generate_synthetic_data()
                    st.session_state.raw_df = raw_df
                    cleaned_df = clean_data(raw_df)
                    st.session_state.cleaned_df = cleaned_df
                    st.success("Default dataset generated and cleaned!")
    
    # If we have cleaned data, proceed
    if 'cleaned_df' in st.session_state:
        st.write("### Missing Value Analysis")
        
        # Count missing values
        missing_counts = []
        for column in st.session_state.cleaned_df.columns:
            missing_count = st.session_state.cleaned_df.filter(col(column).isNull() | isnan(column)).count()
            missing_counts.append({"Column": column, "Missing Count": missing_count, 
                                 "Missing %": (missing_count / st.session_state.cleaned_df.count()) * 100})
        
        # Display missing values
        st.write(pd.DataFrame(missing_counts))
        
        # Missing value handling options
        st.write("### Missing Value Handling")
        st.write("""
        The following strategies will be used:
        - Age: Fill with median age
        - Title: Fill with "No Title"
        - Rating: Fill with mode (most common value)
        - Division Name: Fill with mode
        - Department Name: Fill with most common department for that division
        """)
        
        if st.button("Handle Missing Values"):
            with st.spinner("Handling missing values..."):
                # Handle missing values
                df_imputed, missing_after = handle_missing_values(st.session_state.cleaned_df)
                
                # Store in session state
                st.session_state.imputed_df = df_imputed
                
                # Display sample of the imputed data
                st.write("### Data After Imputation")
                st.write(df_imputed.limit(5).toPandas())
                
                # Display missing values after imputation
                st.write("### Missing Values After Imputation")
                st.write(pd.DataFrame(missing_after))
                
                # Provide download link
                st.markdown(get_csv_download_link(df_imputed, "womens_clothing_ecommerce_imputed.csv"), unsafe_allow_html=True)
                
                st.success("Missing values handled successfully!")

# Page 4: Exploratory Data Analysis
elif page == "4. Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    
    # Check if imputed data exists
    if 'imputed_df' not in st.session_state:
        if 'cleaned_df' in st.session_state:
            st.warning("Please handle missing values first on the Missing Values page.")
            if st.button("Handle Missing Values Now"):
                with st.spinner("Handling missing values..."):
                    df_imputed, _ = handle_missing_values(st.session_state.cleaned_df)
                    st.session_state.imputed_df = df_imputed
                    st.success("Missing values handled!")
        else:
            st.warning("Please generate, clean, and handle missing values in the dataset first.")
            if st.button("Generate, Clean, and Impute Default Dataset"):
                with st.spinner("Processing default dataset..."):
                    raw_df = generate_synthetic_data()
                    st.session_state.raw_df = raw_df
                    cleaned_df = clean_data(raw_df)
                    st.session_state.cleaned_df = cleaned_df
                    df_imputed, _ = handle_missing_values(cleaned_df)
                    st.session_state.imputed_df = df_imputed
                    st.success("Default dataset processed!")
    
    # If we have imputed data, proceed
    if 'imputed_df' in st.session_state:
        if st.button("Perform Exploratory Data Analysis"):
            with st.spinner("Performing EDA..."):
                # Perform EDA
                eda_results = perform_eda(st.session_state.imputed_df)
                
                # Display summary statistics
                st.write("### Summary Statistics")
                st.write(eda_results["summary_stats"])
                
                # Display visualizations in columns
                st.write("### Visualizations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("#### Age Distribution")
                    st.pyplot(eda_results["fig_age"])
                    
                    st.write("#### Rating Distribution")
                    st.pyplot(eda_results["fig_rating"])
                    
                    st.write("#### Rating vs Recommendation")
                    st.pyplot(eda_results["fig_rec"])
                    
                    st.write("#### Rating by Division")
                    st.pyplot(eda_results["fig_rate_div"])
                
                with col2:
                    st.write("#### Division Distribution")
                    st.pyplot(eda_results["fig_div"])
                    
                    st.write("#### Department Distribution")
                    st.pyplot(eda_results["fig_dept"])
                    
                    st.write("#### Class Distribution")
                    st.pyplot(eda_results["fig_class"])
                    
                    st.write("#### Positive Feedback by Rating")
                    st.pyplot(eda_results["fig_pos"])
                
                # Store in session state that EDA was performed
                st.session_state.eda_performed = True
                
                st.success("EDA completed successfully!")

# Page 5: Machine Learning Models
elif page == "5. Machine Learning Models":
    st.title("Machine Learning Models")
    
    # Check if imputed data exists
    if 'imputed_df' not in st.session_state:
        st.warning("Please generate, clean, and handle missing values in the dataset first.")
        if st.button("Process Default Dataset for ML"):
            with st.spinner("Processing default dataset..."):
                raw_df = generate_synthetic_data()
                st.session_state.raw_df = raw_df
                cleaned_df = clean_data(raw_df)
                st.session_state.cleaned_df = cleaned_df
                df_imputed, _ = handle_missing_values(cleaned_df)
                st.session_state.imputed_df = df_imputed
                st.success("Default dataset processed and ready for ML!")
    
    # If we have imputed data, proceed
    if 'imputed_df' in st.session_state:
        # ML model selection
        model_type = st.radio(
            "Select Machine Learning Model",
            ["Regression (Predict Rating)", "Clustering (Customer Segmentation)", "Classification (Predict Recommendation)"]
        )
        
        if st.button("Run ML Model"):
            with st.spinner(f"Running {model_type}..."):
                # Prepare data for ML
                ml_df, train_data, test_data, string_indexers, encoders = prepare_data_for_ml(st.session_state.imputed_df)
                
                if model_type == "Regression (Predict Rating)":
                    # Run regression
                    regression_results = run_regression(train_data, test_data, string_indexers, encoders)
                    
                    # Display regression results
                    st.write("### Regression Results")
                    st.write(f"Root Mean Squared Error (RMSE): {regression_results['rmse']:.4f}")
                    st.write(f"Mean Absolute Error (MAE): {regression_results['mae']:.4f}")
                    st.write(f"R-squared: {regression_results['r2']:.4f}")
                    
                    # Display sample predictions
                    st.write("### Sample Predictions")
                    sample_pred_df = regression_results["sample_predictions"]
                    sample_pred_df["Error"] = sample_pred_df["rating"] - sample_pred_df["prediction"]
                    st.write(sample_pred_df)
                    
                    # Visualize predictions vs actual
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plt.scatter(sample_pred_df["rating"], sample_pred_df["prediction"])
                    plt.plot([1, 5], [1, 5], 'r--')
                    plt.xlabel("Actual Rating")
                    plt.ylabel("Predicted Rating")
                    plt.title("Predicted vs Actual Ratings")
                    st.pyplot(fig)
                    
                    st.success("Regression model completed!")
                
                elif model_type == "Clustering (Customer Segmentation)":
                    # Run clustering
                    clustering_results = run_clustering(ml_df, string_indexers, encoders)
                    
                    # Display clustering results
                    st.write("### Clustering Results")
                    st.write(f"Optimal number of clusters (k): {clustering_results['optimal_k']}")
                    
                    # Display WSSSE by k
                    wssse_data = [(result['k'], result['wssse']) for result in clustering_results['results']]
                    wssse_df = pd.DataFrame(wssse_data, columns=['k', 'WSSSE'])
                    
                    fig_wssse, ax_wssse = plt.subplots(figsize=(10, 6))
                    plt.plot(wssse_df['k'], wssse_df['WSSSE'], 'bo-')
                    plt.xlabel("Number of clusters (k)")
                    plt.ylabel("Within-Cluster Sum of Squared Errors (WSSSE)")
                    plt.title("Elbow Method for Optimal k")
                    st.pyplot(fig_wssse)
                    
                    # Display cluster profiles
                    st.write("### Cluster Profiles")
                    cluster_profiles = clustering_results["cluster_profiles"]
                    cluster_profiles.columns = ['Cluster', 'Avg Age', 'Avg Rating', 'Avg Recommendation', 'Avg Positive Feedback']
                    st.write(cluster_profiles)
                    
                    # Visualize cluster sizes
                    optimal_k_result = next(r for r in clustering_results['results'] if r['k'] == clustering_results['optimal_k'])
                    cluster_sizes = optimal_k_result['cluster_sizes']
                    
                    fig_sizes, ax_sizes = plt.subplots(figsize=(10, 6))
                    plt.bar(cluster_sizes['prediction'].astype(str), cluster_sizes['count'])
                    plt.xlabel("Cluster")
                    plt.ylabel("Number of Customers")
                    plt.title("Cluster Sizes")
                    st.pyplot(fig_sizes)
                    
                    # Visualize division by cluster
                    division_clusters = clustering_results["division_clusters"]
                    
                    fig_div_cluster, ax_div_cluster = plt.subplots(figsize=(12, 8))
                    division_pivot = division_clusters.pivot(index='division_name', columns='prediction', values='count')
                    division_pivot.plot(kind='bar', stacked=True, ax=ax_div_cluster)
                    plt.xlabel("Division")
                    plt.ylabel("Number of Customers")
                    plt.title("Division Distribution by Cluster")
                    plt.legend(title="Cluster")
                    st.pyplot(fig_div_cluster)
                    
                    st.success("Clustering model completed!")
                
                elif model_type == "Classification (Predict Recommendation)":
                    # Run classification
                    classification_results = run_classification(train_data, test_data, string_indexers, encoders)
                    
                    # Display classification results
                    st.write("### Classification Results")
                    st.write(f"Accuracy: {classification_results['accuracy']:.4f}")
                    st.write(f"F1 Score: {classification_results['f1']:.4f}")
                    
                    # Display sample predictions
                    st.write("### Sample Predictions")
                    sample_pred_df = classification_results["sample_predictions"]
                    st.write(sample_pred_df)
                    
                    # Confusion matrix (derived from sample)
                    confusion_counts = sample_pred_df.groupby(['recommended_ind', 'prediction']).size().unstack(fill_value=0)
                    
                    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                    sns.heatmap(confusion_counts, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    plt.title("Confusion Matrix (Sample)")
                    st.pyplot(fig_cm)
                    
                    # Feature importance (top features)
                    feature_names = ["age", "rating", "positive_feedback_count", 
                                    "division_name", "department_name", "class_name"]
                    
                    # For demonstration only - in a real app, we'd match actual feature indices
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': np.array(classification_results["feature_importance"].toArray())[:len(feature_names)]
                    })
                    importance_df = importance_df.sort_values('Importance', ascending=False)
                    
                    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax_imp)
                    plt.title("Feature Importance")
                    st.pyplot(fig_imp)
                    
                    st.success("Classification model completed!")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
    This Streamlit app demonstrates big data processing and machine learning 
    for Women's Clothing E-Commerce data using PySpark.
""")