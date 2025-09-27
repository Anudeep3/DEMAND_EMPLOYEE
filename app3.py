import streamlit as st
import pandas as pd
from io import BytesIO

# ----------------------------
# CLEANING FUNCTIONS
# ----------------------------

def clean_demand(df):
    cols = [
        "Request profile number", "Request Profile Status", "Customer ID", "Customer Name",
        "Service Line", "Practice (Cluster)", "IBU", "Project Bill type", "Grade",
        "Billing requirement", "Mandatory skill 1", "Mandatory skill 2", "Mandatory skill 3",
        "Onsite or Offshore", "Office Country", "Office City", "Client Interview",
        "Deal breaker skill", "Customer Group ID", "Pricing Grade", "Service Line ID",
        "Practice ID", "IBU ID"
    ]
    df = df[cols].copy()

    # Remove unwanted statuses
    bad_status = ["Cancelled", "Declined", "Fulfilled", "Withdrawn"]
    df = df[~df["Request Profile Status"].isin(bad_status)]

    # Remove un-billed
    df = df[df["Billing requirement"].fillna("").str.strip().str.lower() != "un-billed"]

    # Combine skills
    df["demand skills"] = (
        df[["Mandatory skill 1","Mandatory skill 2","Mandatory skill 3","Deal breaker skill"]]
        .fillna("")
        .apply(lambda row: ",".join([str(x).strip() for x in row if str(x).strip() != ""]), axis=1)
    )

    # Deduplicate
    def dedup(skills):
        seen, uniq = set(), []
        for s in str(skills).split(","):
            s = s.strip()
            if s and s not in seen:
                seen.add(s); uniq.append(s)
        return ",".join(uniq)

    df["demand skills"] = df["demand skills"].apply(dedup)
    return df


def clean_employee(df):
    cols = [
        "EMPLID","EMPLOYEE_IBU","EMPLOYEE_IBU_DESCRIPTION","EMPLOYEE_SERVICE_LINE","EMPLOYEE_SERVICE_LINE_DESC",
        "BAND","CURRENT_COUNTRY","CURRENT_LOCATION_CITY","ONSITE_OFFSHORE",
        "PROJECT_ID","PROJECT_PRICING_TYPE","PROJECT_IBU","PROJECT_TYPE",
        "CUSTOMER_ID","CUSTOMER_NAME","CUSTOMER_GROUP","CUSTOMER_GROUP_NAME","BW_CATEGORY",
        "TECH_SKILL_1","TECH_SKILL_2","TECH_SKILL_3","TECH_SKILL_4"
    ]
    df = df[cols].copy()

    skill_cols = ["TECH_SKILL_1","TECH_SKILL_2","TECH_SKILL_3","TECH_SKILL_4"]

    # Combine skills
    df["employee_skills"] = (
        df[skill_cols].fillna("")
        .apply(lambda r: ",".join([str(x).strip() for x in r if str(x).strip() != ""]), axis=1)
    )

    # Remove levels
    for lvl in ["/Expert","/Advanced","/Intermediate","/Foundational"]:
        df["employee_skills"] = df["employee_skills"].str.replace(lvl, "", regex=False)

    # Dedup
    def dedup(skills):
        seen, uniq = set(), []
        for s in str(skills).split(","):
            s = s.strip()
            if s and s not in seen:
                seen.add(s); uniq.append(s)
        return ",".join(uniq)

    df["employee_skills"] = df["employee_skills"].apply(dedup)
    df["EMPLID_str"] = df["EMPLID"].astype(str).str.strip()
    return df


# ----------------------------
# MATCHING + SUMMARY
# ----------------------------

def match_demand_employee(df_demand, df_employee):
    df_demand["demand_skills_list"] = df_demand["demand skills"].fillna("").str.lower().str.split(",")
    df_employee["employee_skills_list"] = df_employee["employee_skills"].fillna("").str.lower().str.split(",")

    df_demand_exploded = df_demand.explode("demand_skills_list").rename(columns={"demand_skills_list": "skill"})
    df_employee_exploded = df_employee.explode("employee_skills_list").rename(columns={"employee_skills_list": "skill"})

    df_demand_exploded["skill"] = df_demand_exploded["skill"].str.strip()
    df_employee_exploded["skill"] = df_employee_exploded["skill"].str.strip()

    df_match = pd.merge(df_demand_exploded, df_employee_exploded, on="skill", how="inner")

    demand_skill_counts = df_demand_exploded.groupby("Request profile number")["skill"].nunique().rename("total_demand_skills")

    df_result = (
        df_match.groupby(["Request profile number", "EMPLID"])
        .agg(
            matched_skills=("skill", lambda x: ",".join(sorted(set(x)))),
            matched_count=("skill", "nunique")
        )
        .reset_index()
        .merge(demand_skill_counts, on="Request profile number", how="left")
    )

    df_result["Match %"] = round((df_result["matched_count"] / df_result["total_demand_skills"]) * 100, 2)
    return df_result


def demand_employee_summary(df):
    grouped = df.groupby("Request profile number").apply(
        lambda g: ",".join([f"{row.EMPLID}({row['Match %']}%)" for _,row in g.iterrows()])
    ).reset_index(name="Associates")
    return grouped


# ----------------------------
# FILTER LOGIC
# ----------------------------

def apply_filters(df, options):
    band_order = ["U1","U2","U3","U4","P1","P2","E1","E2","E3"]
    band_index = {b: i for i, b in enumerate(band_order)}

    def normalize_lower(x): return str(x).strip().lower() if pd.notna(x) else ""
    def normalize_upper(x): return str(x).strip().upper() if pd.notna(x) else ""

    df = df.copy()
    df["ONSITE_OFFSHORE_norm"] = df["ONSITE_OFFSHORE"].apply(normalize_lower)
    df["Onsite_Offshore_norm"] = df["Onsite or Offshore"].apply(normalize_lower)
    df["BAND_norm"] = df["BAND"].apply(normalize_upper)
    df["Grade_norm"] = df["Grade"].apply(normalize_upper)

    if 1 in options:  # offshore only
        df = df[(df["ONSITE_OFFSHORE_norm"]=="offshore") & (df["Onsite_Offshore_norm"]=="offshore")]

    if 2 in options:  # onsite only
        df = df[(df["ONSITE_OFFSHORE_norm"]=="onsite") & (df["Onsite_Offshore_norm"]=="onsite")]

    if 3 in options:  # both onsite/offshore match
        df = df[df["ONSITE_OFFSHORE_norm"]==df["Onsite_Offshore_norm"]]

    if 4 in options:  # band logic
        def band_logic(r):
            e, d = r["BAND_norm"], r["Grade_norm"]
            if e not in band_index or d not in band_index: return False
            e_idx = band_index[e]
            valid_demand = [band_order[e_idx]]
            if e_idx > 0: valid_demand.append(band_order[e_idx-1])
            return d in valid_demand
        df = df[df.apply(band_logic, axis=1)]

    return df


# ----------------------------
# STARVATION LOGIC
# ----------------------------

def starvation_logic(df_summary):
    rows=[]
    for _, row in df_summary.iterrows():
        demand=row["Request profile number"]
        associates=str(row["Associates"]).split(",")
        for assoc in associates:
            assoc=assoc.strip()
            if assoc:
                emp_id=assoc.split("(")[0]
                match_pct=float(assoc.split("(")[1].replace("%)",""))
                rows.append([demand,int(emp_id),match_pct,row["Associates"]])
    df_expanded=pd.DataFrame(rows,columns=["Demand","Employee","Match_%","All_Employees"])

    assignments={}
    used_emps=set()

    # Step 1: Unique Fix Loop
    while True:
        demand_counts=df_expanded.groupby("Demand")["Employee"].nunique()
        uniques=demand_counts[demand_counts==1].index
        if uniques.empty: break
        unique_rows=df_expanded[df_expanded["Demand"].isin(uniques)].drop_duplicates("Demand")
        for _,row in unique_rows.iterrows():
            d,e=row["Demand"],row["Employee"]
            if d not in assignments:
                assignments[d]=(e,row["Match_%"],None,row["All_Employees"],"Unique Fix")
                used_emps.add(e)
        df_expanded=df_expanded[~df_expanded["Employee"].isin(used_emps)]

    # Step 2: Score Remaining
    if not df_expanded.empty:
        emp_demand_counts=df_expanded.groupby("Employee")["Demand"].nunique().to_dict()
        df_expanded["num_factor"]=df_expanded["Employee"].map(emp_demand_counts)
        w1,w2=0.7,0.3
        df_expanded["Score"]=w1*df_expanded["Match_%"]+w2*(1/df_expanded["num_factor"])
        df_expanded=df_expanded.sort_values("Score",ascending=False)

        # Assign uniquely first
        for _,row in df_expanded.iterrows():
            d,e=row["Demand"],row["Employee"]
            if d not in assignments and e not in used_emps:
                assignments[d]=(e,row["Match_%"],row["Score"],row["All_Employees"],"Score Unique")
                used_emps.add(e)

        # Assign remaining (reuse with *)
        for _,row in df_expanded.iterrows():
            d,e=row["Demand"],row["Employee"]
            if d not in assignments:
                assignments[d]=(str(e)+"*",row["Match_%"],row["Score"],row["All_Employees"],"Score Reuse")

    final_df=pd.DataFrame(
        [(d,emp,match,score,all_emps,method) for d,(emp,match,score,all_emps,method) in assignments.items()],
        columns=["Demand","Assigned_Employee","Match_%","Score","All_Employees","Method"]
    )
    return final_df


# ----------------------------
# STREAMLIT APP
# ----------------------------

st.title("Demand ↔ Employee Matching & Assignment")

# Upload files
demand_file=st.file_uploader("Upload Demand File",type=["xls","xlsx"])
employee_file=st.file_uploader("Upload Employee File",type=["xls","xlsx"])

if demand_file and employee_file:
    df_demand=clean_demand(pd.read_excel(demand_file))
    df_employee=clean_employee(pd.read_excel(employee_file))

    # Matching
    df_matches=match_demand_employee(df_demand,df_employee)

    # Merge details back (first merge)
    df_matches=df_matches.merge(df_demand,on="Request profile number",how="left").merge(df_employee,on="EMPLID",how="left")
    st.write("Sample Matches",df_matches.head())

    # Filter options
    st.subheader("Choose Filter Options")
    st.markdown("""
    **Available Filters:**  
    - `1` → Offshore Only  
    - `2` → Onsite Only  
    - `3` → Both Onsite/Offshore Match  
    - `4` → Band Logic  
    """)
    options = st.text_input("Enter filter options (e.g., 1,2,3,4):")
    if options:
        options = [int(x.strip()) for x in options.split(",") if x.strip().isdigit()]
        df_matches = apply_filters(df_matches,options)
        st.write("Filtered Matches",df_matches.head())

    # Summary after filter
    df_summary=demand_employee_summary(df_matches)
    st.write("Demand–Employee Summary",df_summary.head())

    # Starvation Logic
    df_starvation=starvation_logic(df_summary)

    # Merge details back again
    df_starvation["Assigned_Employee_key"]=df_starvation["Assigned_Employee"].astype(str).str.replace("*","",regex=False).str.strip()
    df_employee["EMPLID_str"]=df_employee["EMPLID"].astype(str).str.strip()
    df_final = df_starvation.merge(df_demand, left_on="Demand", right_on="Request profile number", how="left") \
                            .merge(df_employee, left_on="Assigned_Employee_key", right_on="EMPLID_str", how="left")

    st.write("Final Assignments with Details",df_final.head())

    # Download
    out=BytesIO()
    df_final.to_excel(out,index=False)
    st.download_button("Download Final Excel",out.getvalue(),file_name="Final_Assignments.xlsx",mime="application/vnd.ms-excel")
