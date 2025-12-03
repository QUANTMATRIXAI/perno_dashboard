# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import seaborn as sns

# st.set_page_config(page_title="Interactive Data Explorer", layout="wide")

# # --- Sidebar Navigation ---
# page = st.sidebar.radio("📌 Navigation", ["Upload Data", "Explore Data", "Charts", "Pivot & Stats"])
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import duckdb
# from pivottablejs import pivot_ui
# import streamlit.components.v1 as components
# import tempfile

# # st.set_page_config(page_title="Fast Explorer", layout="wide")

# ############### PERFORMANCE BOOST FUNCTIONS ###############

# @st.cache_data(show_spinner=True)
# def load_data(file):
#     """Load and convert data to parquet for future reuse."""
#     if file.name.endswith(".csv"):
#         df = pd.read_csv(file)
#     else:
#         df = pd.read_excel(file)

#     # Convert to parquet for extremely fast reload next time
#     parquet_file = "cache.parquet"
#     df.to_parquet(parquet_file, index=False)
#     return df, parquet_file


# @st.cache_data(show_spinner=False)
# def fast_filter(parquet_file, query=None):
#     """Use DuckDB for extremely fast SELECT queries on large datasets."""
#     if query:
#         return duckdb.query(f"SELECT * FROM parquet_scan('{parquet_file}') WHERE {query}").to_df()
#     return duckdb.query(f"SELECT * FROM parquet_scan('{parquet_file}')").to_df()

# #############################################################

# st.title("⚡ Ultra Fast Interactive Dashboard")

# uploaded_file = st.file_uploader("Upload large dataset (CSV/Excel)", type=["csv", "xlsx"])

# if uploaded_file and "df" not in st.session_state:

#     with st.spinner("Loading data..."):
#         df, parquet_path = load_data(uploaded_file)

#     st.session_state.df = df
#     st.session_state.parquet = parquet_path
#     st.success("Data loaded and cached successfully!")


# # --- Load Data if Available ---
# df = st.session_state.df

# if df is not None:

#     # Sidebar Filters
#     st.sidebar.write("### 🔽 Filters")

#     # Detect Revised Seg column
#     seg_col = None
#     for col in df.columns:
#         if "Revised Seg" in col:
#             seg_col = col
#             break

#     # Revised Seg Filter (if found)
#     if seg_col:
#         selected_seg = st.sidebar.multiselect(
#             f"Filter by {seg_col}", df[seg_col].dropna().unique(), default=df[seg_col].dropna().unique()
#         )
#         df = df[df[seg_col].isin(selected_seg)]

#     numeric_cols = df.select_dtypes(include="number").columns.tolist()
#     categorical_cols = df.select_dtypes(include="object").columns.tolist()

#     # --- Page 2: Explore Data ---
#     if page == "Explore Data":
#         st.title("🔍 Explore Data")

#         st.subheader("Filtered Data Table")
#         st.dataframe(df, use_container_width=True)

#         st.write("### Correlation Heatmap")
#         if len(numeric_cols) > 1:
#             fig = px.imshow(df[numeric_cols].corr(), text_auto=True, aspect="auto")
#             st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.warning("Need at least 2 numeric columns for correlation heatmap.")

#     # --- Page 3: Charts ---
#     elif page == "Charts":
#         st.title("📈 Interactive Charts")

#         col1, col2 = st.columns(2)
#         x_axis = col1.selectbox("Select X-axis", df.columns)
#         y_axis = col2.selectbox("Select Y-axis", numeric_cols)

#         chart_type = st.radio("Select Chart Type", ["Bar", "Line", "Scatter", "Histogram"])

#         if chart_type == "Bar":
#             fig = px.bar(df, x=x_axis, y=y_axis, color=seg_col if seg_col else None)
#         elif chart_type == "Line":
#             fig = px.line(df, x=x_axis, y=y_axis, color=seg_col if seg_col else None)
#         elif chart_type == "Scatter":
#             fig = px.scatter(df, x=x_axis, y=y_axis, color=seg_col if seg_col else None, trendline="ols")
#         else:
#             fig = px.histogram(df, x=y_axis, color=seg_col if seg_col else None)

#         st.plotly_chart(fig, use_container_width=True)

#     # --- Page 4: Pivot & Stats ---
#     elif page == "Pivot & Stats":
#         st.title("📊 Pivot Analysis & KPIs")

#         # with st.expander("Create Pivot Table"):
#         #     p1, p2, p3 = st.columns(3)

#         #     rows = p1.multiselect("Row Category", categorical_cols)
#         #     values = p2.selectbox("Value Column", numeric_cols)
#         #     agg = p3.selectbox("Aggregation", ["mean", "sum", "median", "count"])

#         #     if rows and values:
#         #         pivot_table = df.pivot_table(index=rows, values=values, aggfunc=agg, sort=True)
#         #         st.dataframe(pivot_table, use_container_width=True)
#         from pivottablejs import pivot_ui
#         import streamlit.components.v1 as components

#         st.subheader("📊 Excel-Style Pivot Table")

#         with st.expander("🔧 Build Pivot Table (Interactive Like Excel)", expanded=True):

#             st.write("Select fields below and scroll down to interactive pivot view.")

#             # ► Column selectors
#             c1, c2, c3 = st.columns(3)

#             row_fields = c1.multiselect("Rows", categorical_cols, default=None)
#             col_fields = c2.multiselect("Columns", categorical_cols, default=None)
#             value_fields = c3.multiselect("Values", numeric_cols, default=None)

#             agg_options = ["sum", "mean", "median", "count", "min", "max", "std"]
#             agg_func = st.selectbox("Aggregation Function", agg_options, index=0)

#             show_totals = st.checkbox("Show Row/Column Totals", value=True)
            
#             # Pivot Execution
#             if value_fields:
#                 try:
#                     pivot_table = pd.pivot_table(
#                         df, 
#                         index=row_fields if row_fields else None,
#                         columns=col_fields if col_fields else None,
#                         values=value_fields,
#                         aggfunc=agg_func,
#                         margins=show_totals,
#                         margins_name="Total"
#                     )

#                     st.write("📁 Pivot Table Output (Static View)")
#                     st.dataframe(pivot_table, use_container_width=True)

#                     # Export button
#                     pivot_csv = pivot_table.to_csv().encode('utf-8')
#                     st.download_button(
#                         label="⬇ Download Pivot as CSV",
#                         data=pivot_csv,
#                         file_name="pivot_table.csv",
#                         mime="text/csv"
#                     )

#                 except Exception as e:
#                     st.error(f"⚠️ Pivot Error: {e}")

#             # # Interactive version (browser-powered Excel-like pivot)
#             # st.write("🧊 Interactive Pivot (Drag & Drop):")

#             # # Generate temporary pivot UI file
#             # try:
#             #     with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
#             #         pivot_ui(df, outfile=f.name)

#             #         # Display interactive pivot table
#             #         components.html(open(f.name, 'r').read(), height=650, scrolling=True)

#             #     st.success("Interactive Pivot Loaded!")
            
#             # except Exception as e:
#             #     st.error("⚠️ Could not load pivot table:")
#             #     st.write(e)


#         st.subheader("📌 Summary Statistics")
#         st.dataframe(df.describe(), use_container_width=True)

# else:
#     st.warning("📌 Please upload a dataset from the sidebar to continue.")

import streamlit as st
import plotly.express as px
import pandas as pd
import requests
import json

st.set_page_config(layout="wide")

# ---- Load GeoJSON Online ----
@st.cache_data
def load_geojson():
    url = "https://raw.githubusercontent.com/datameet/maps/master/States/Admin2/india_states.geojson"
    response = requests.get(url)
    india_geo = response.json()   # this now works
    return india_geo

india_geo = load_geojson()

# Extract state names
all_states = sorted([feature["properties"]["st_nm"] for feature in india_geo["features"]])

# ---- User selection ----
selected_states = st.multiselect(
    "Select States to Highlight",
    options=all_states,
    default=[]
)

# Create dataframe
df = pd.DataFrame({"state": all_states})
df["highlight"] = df["state"].apply(lambda x: 1 if x in selected_states else 0)

# ---- Plot Map ----
fig = px.choropleth(
    df,
    geojson=india_geo,
    featureidkey="properties.st_nm",
    locations="state",
    color="highlight",
    color_continuous_scale=["lightgray", "orange"]
)
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(coloraxis_showscale=False, title="India State Selector")

st.plotly_chart(fig, use_container_width=True)
