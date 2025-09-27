import streamlit as st
import pandas as pd

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

    bad_status = ["Cancelled", "Declined", "Fulfilled", "Withdrawn"]
    df = df[~df["Request Profile Status"].isin(bad_status)]

    df = df[df["Billing requirement"].fillna("").str.strip().str.lower() != "un-billed"]

    df["demand skills"] = (
        df[["Mandatory skill 1","Mandatory skill 2","Mandatory skill 3","Deal breaker skill"]]
        .fillna("")
        .apply(lambda row: ",".join([str(x).strip() for x in row if str(x).strip() != ""]), axis=1)
    )

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
    cols = ["EMPLID","Employee Name","ONSITE_OFFSHORE","BAND","Grade",
            "Office Country","Office City","TECH_SKILL_1","TECH_SKILL_2","TECH_SKILL_3","TECH_SKILL_4"]
    df = df[cols].copy()

    skill_cols = ["TECH_SKILL_1","TECH_SKILL_2","TECH_SKILL_3","TECH_SKILL_4"]

    df["employee_skills"] = (
        df[skill_cols].fillna("")
        .apply(lambda r: ",".join([str(x).strip() for x in r if str(x).strip() != ""]), axis=1)
    )

    for lvl in ["/Expert","/Advanced","/Intermediate","/Foundational"]:
        df["employee_skills"] = df["employee_skills"].str.replace(lvl, "", regex=False)

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
# MATCHING
# ----------------------------

def match_demand_employee(df_demand, df_employee):
    df_demand["demand_skills_list"] = df_demand["demand skills"].fillna("").str.lower().str.split(",")
    df_employee["employee_skills_list"] = df_employee["employee_skills"].fillna("").str.lower().str.split(",")

    dmd = df_demand.explode("demand_skills_list").rename(columns={"demand_skills_list":"skill"})
    emp = df_employee.explode("employee_skills_list").rename(columns={"employee_skills_list":"skill"})

    dmd["skill"] = dmd["skill"].str.strip()
    emp["skill"] = emp["skill"].str.strip()

    dmd = dmd[dmd["skill"] != ""]
    emp = emp[emp["skill"] != ""]

    merged = pd.merge(dmd, emp, on="skill", how="inner")

    demand_skill_counts = dmd.groupby("Request profile number")["skill"].nunique().rename("total_demand_skills")

    res = (
        merged.groupby(["Request profile number","EMPLID"], as_index=False)
        .agg(
            matched_skills=("skill", lambda x: ",".join(sorted(set(x)))),
            matched_count=("skill", "nunique")
        )
    )
    res = res.merge(demand_skill_counts, on="Request profile number", how="left")
    res["Match %"] = round((res["matched_count"] / res["total_demand_skills"]) * 100, 2)
    res["EMPLID_str"] = res["EMPLID"].astype(str).str.strip()
    return res

# ----------------------------
# FILTER LOGIC (Colab style)
# ----------------------------

def normalize_lower(v):
    if pd.isna(v): return None
    return str(v).strip().lower()

def normalize_upper(v):
    if pd.isna(v): return None
    return str(v).strip().upper()

band_order = ["U1","U2","U3","U4","P1","P2","E1","E2","E3"]
band_index = {b:i for i,b in enumerate(band_order)}

def band_logic(row):
    band = normalize_upper(row["BAND"])
    grade = normalize_upper(row["Grade"])
    if not band or not grade: return False
    if band not in band_index or grade not in band_index: return False
    idx = band_index[band]
    valid = [band_order[idx]]
    if idx > 0: valid.append(band_order[idx-1])
    return grade in valid

def filter_data(df, options):
    df = df.copy()
    df["ONSITE_OFFSHORE_norm"] = df["ONSITE_OFFSHORE"].apply(normalize_lower)
    df["Onsite_Offshore_norm"] = df["Onsite or Offshore"].apply(normalize_lower)

    if 1 in options:
        df = df[(df["ONSITE_OFFSHORE_norm"]=="offshore") & (df["Onsite_Offshore_norm"]=="offshore")]
    if 2 in options:
        df = df[(df["ONSITE_OFFSHORE_norm"]=="onsite") & (df["Onsite_Offshore_norm"]=="onsite")]
    if 3 in options:
        df = df[df["ONSITE_OFFSHORE_norm"]==df["Onsite_Offshore_norm"]]
    if 4 in options:
        df = df[df.apply(band_logic, axis=1)]
    return df

# ----------------------------
# SUMMARY + STARVATION
# ----------------------------

def demand_employee_summary(df):
    grouped = df.groupby("Request profile number").apply(
        lambda g: ",".join([f"{row.EMPLID}({row['Match %']}%)" for _,row in g.iterrows()])
    ).reset_index(name="Associates")
    grouped["Employee_Count"] = grouped["Associates"].apply(lambda x: len(x.split(",")) if x else 0)
    return grouped

def starvation_logic(df_summary):
    rows=[]
    for _, row in df_summary.iterrows():
        demand=row["Request profile number"]
        for assoc in str(row["Associates"]).split(","):
            assoc=assoc.strip()
            if not assoc: continue
            emp_id=assoc.split("(")[0].strip()
            match=float(assoc.split("(")[1].replace("%)",""))
            rows.append([demand, emp_id, match, row["Associates"]])
    df_exp = pd.DataFrame(rows, columns=["Demand","Employee","Match_%","All_Employees"])
    if df_exp.empty: return pd.DataFrame(columns=["Demand","Assigned_Employee","Match_%","Score","All_Employees","Method"])

    assignments={}
    used=set()

    # Unique fix
    while True:
        counts=df_exp.groupby("Demand")["Employee"].nunique()
        uniques=counts[counts==1].index
        if uniques.empty: break
        uniq_rows=df_exp[df_exp["Demand"].isin(uniques)].drop_duplicates("Demand")
        for _,r in uniq_rows.iterrows():
            d,e=r["Demand"],str(r["Employee"])
            if d not in assignments:
                assignments[d]=(e,r["Match_%"],None,r["All_Employees"],"Unique Fix")
                used.add(e)
        df_exp=df_exp[~df_exp["Employee"].astype(str).isin(used)]

    if not df_exp.empty:
        counts=df_exp.groupby("Employee")["Demand"].nunique().to_dict()
        df_exp["num_factor"]=df_exp["Employee"].map(counts)
        w1,w2=0.7,0.3
        df_exp["Score"]=w1*df_exp["Match_%"]+w2*(1/df_exp["num_factor"])
        df_exp=df_exp.sort_values("Score",ascending=False)

        for _,r in df_exp.iterrows():
            d,e=r["Demand"],str(r["Employee"])
            if d not in assignments and e not in used:
                assignments[d]=(e,r["Match_%"],r["Score"],r["All_Employees"],"Score Unique")
                used.add(e)

        for _,r in df_exp.iterrows():
            d,e=r["Demand"],str(r["Employee"])
            if d not in assignments:
                assignments[d]=(e+"*",r["Match_%"],r["Score"],r["All_Employees"],"Score Reuse")

    final=pd.DataFrame(
        [(d,e,m,s,a,mth) for d,(e,m,s,a,mth) in assignments.items()],
        columns=["Demand","Assigned_Employee","Match_%","Score","All_Employees","Method"]
    )
    final["Assigned_Employee_key"]=final["Assigned_Employee"].str.replace("*","",regex=False).str.strip()
    return final

# ----------------------------
# STREAMLIT APP
# ----------------------------

st.title("Demand ↔ Employee Matching & Assignment")

demand_file=st.file_uploader("Upload Demand File", type=["xls","xlsx"])
employee_file=st.file_uploader("Upload Employee File", type=["xls","xlsx"])

if demand_file and employee_file:
    df_demand=clean_demand(pd.read_excel(demand_file))
    df_employee=clean_employee(pd.read_excel(employee_file))

    # Matching
    df_matches=match_demand_employee(df_demand,df_employee)

    # Merge after first match
    df_matches_full=df_matches.merge(df_demand,on="Request profile number",how="left") \
                              .merge(df_employee,on="EMPLID",how="left")
    st.subheader("First Summary with Details")
    st.write(df_matches_full.head())

    # Filter
    st.subheader("Apply Filters")
    st.markdown("""
    - `1` → Offshore Only  
    - `2` → Onsite Only  
    - `3` → Both Onsite/Offshore Match  
    - `4` → Band Logic  
    """)
    options_text=st.text_input("Enter filter options (e.g., 1,2,3,4):","")
    options=[int(x.strip()) for x in options_text.split(",") if x.strip().isdigit()]
    if options:
        df_filtered=filter_data(df_matches_full,options)
        st.write("Rows before:",len(df_matches_full))
        st.write("Rows after:",len(df_filtered))
        st.write("Dropped:",len(df_matches_full)-len(df_filtered))
        st.subheader("Filtered Matches")
        st.write(df_filtered.head())
    else:
        df_filtered=df_matches_full.copy()

    # Summary
    df_summary=demand_employee_summary(df_filtered)
    st.subheader("Demand–Employee Summary (After Filter)")
    st.write(df_summary.head())

    # Starvation
    df_starvation=starvation_logic(df_summary)
    st.subheader("Starvation Assignments")
    st.write(df_starvation.head())

    # Merge final details
    df_final=df_starvation.merge(df_demand,left_on="Demand",right_on="Request profile number",how="left") \
                          .merge(df_employee,left_on="Assigned_Employee_key",right_on="EMPLID_str",how="left")

    st.subheader("Final Assignments with Demand & Employee Details")
    st.write(df_final.head())

    # Download
    out="Final_Assignments_with_Details.xlsx"
    df_final.to_excel(out,index=False)
    with open(out,"rb") as f:
        st.download_button("Download Final Excel",f,out)
