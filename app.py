import streamlit as st
import pandas as pd

# ---------------- UTILS ----------------
def normalize_lower(val):
    if pd.isna(val): return None
    return str(val).strip().lower()

def normalize_upper(val):
    if pd.isna(val): return None
    return str(val).strip().upper()

band_order = ["U1","U2","U3","U4","P1","P2","E1","E2","E3"]
band_index = {b: i for i, b in enumerate(band_order)}

def band_logic(row):
    band = normalize_upper(row["BAND"])
    grade = normalize_upper(row["Grade"])
    if not band or not grade: return False
    if band not in band_index or grade not in band_index: return False
    idx = band_index[band]
    valid_grades = [band_order[idx]]
    if idx + 1 < len(band_order):
        valid_grades.append(band_order[idx + 1])
    return grade in valid_grades

def filter_data(df, options):
    df["ONSITE_OFFSHORE_norm"] = df["ONSITE_OFFSHORE"].apply(normalize_lower)
    df["Onsite_Offshore_norm"] = df["Onsite or Offshore"].apply(normalize_lower)
    filtered = df.copy()

    if 1 in options:  # offshore only
        filtered = filtered[
            (filtered["ONSITE_OFFSHORE_norm"]=="offshore") &
            (filtered["Onsite_Offshore_norm"]=="offshore")
        ]
    if 2 in options:  # onsite only
        filtered = filtered[
            (filtered["ONSITE_OFFSHORE_norm"]=="onsite") &
            (filtered["Onsite_Offshore_norm"]=="onsite")
        ]
    if 3 in options:  # both onsite/offshore
        filtered = filtered[filtered["ONSITE_OFFSHORE_norm"]==filtered["Onsite_Offshore_norm"]]
    if 4 in options:  # band logic
        mask = filtered.apply(band_logic, axis=1)
        filtered = filtered[mask]

    return filtered

# ---- demand cleaning
def clean_demand(df):
    bad_status=["Cancelled","Declined","Fulfilled","Withdrawn"]
    df=df[~df["Request Profile Status"].isin(bad_status)]
    df=df[df["Billing requirement"].str.strip().str.lower()!="un-billed"]
    df["demand skills"]=(
        df[["Mandatory skill 1","Mandatory skill 2","Mandatory skill 3","Deal breaker skill"]]
        .fillna("")
        .apply(lambda row: ",".join([str(x).strip() for x in row if x!=""]), axis=1)
    )
    def dedup(s):
        seen, uniq=set(),[]
        for x in s.split(","):
            x=x.strip()
            if x and x not in seen: uniq.append(x); seen.add(x)
        return ",".join(uniq)
    df["demand skills"]=df["demand skills"].apply(dedup)
    return df

# ---- employee cleaning
def clean_employee(df):
    skill_cols=["TECH_SKILL_1","TECH_SKILL_2","TECH_SKILL_3","TECH_SKILL_4"]
    df["employee_skills"]=df[skill_cols].fillna("").apply(
        lambda row: ",".join([str(x).strip() for x in row if x!=""]), axis=1
    )
    for lvl in ["/Expert","/Advanced","/Intermediate","/Foundational"]:
        df["employee_skills"]=df["employee_skills"].str.replace(lvl,"",regex=False)
    def dedup(s):
        seen, uniq=set(),[]
        for x in s.split(","):
            x=x.strip()
            if x and x not in seen: uniq.append(x); seen.add(x)
        return ",".join(uniq)
    df["employee_skills"]=df["employee_skills"].apply(dedup)
    return df

# ---- matching
def match_demand_employee(df_demand, df_employee):
    df_demand["demand_skills_list"]=df_demand["demand skills"].fillna("").str.lower().str.split(",")
    df_employee["employee_skills_list"]=df_employee["employee_skills"].fillna("").str.lower().str.split(",")
    dmd=df_demand.explode("demand_skills_list").rename(columns={"demand_skills_list":"skill"})
    emp=df_employee.explode("employee_skills_list").rename(columns={"employee_skills_list":"skill"})
    dmd["skill"]=dmd["skill"].str.strip(); emp["skill"]=emp["skill"].str.strip()
    merged=pd.merge(dmd,emp,on="skill",how="inner")
    demand_skill_counts=dmd.groupby("Request profile number")["skill"].nunique().rename("total_demand_skills")
    res=(merged.groupby(["Request profile number","EMPID"])
         .agg(matched_skills=("skill",lambda x: ",".join(sorted(set(x)))),
              matched_count=("skill","nunique"))
         .reset_index()
         .merge(demand_skill_counts,on="Request profile number",how="left"))
    res["Match %"]=round((res["matched_count"]/res["total_demand_skills"])*100,2)
    return res

# ---- summary
def demand_employee_summary(df):
    grouped=df.groupby("Request profile number").apply(
        lambda g: ",".join([f"{row.EMPID}({row['Match %']}%)" for _,row in g.iterrows()])
    ).reset_index(name="Associates")
    grouped["Employee_Count"]=grouped["Associates"].apply(lambda x: len(x.split(",")) if x else 0)
    return grouped

# ---- starvation
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
    assignments={}; used_emps=set()
    while True:
        demand_counts=df_expanded.groupby("Demand")["Employee"].nunique()
        uniques=demand_counts[demand_counts==1].index
        if uniques.empty: break
        unique_rows=df_expanded[df_expanded["Demand"].isin(uniques)].drop_duplicates("Demand")
        for _,row in unique_rows.iterrows():
            d,e=row["Demand"],row["Employee"]
            if d not in assignments:
                assignments[d]=(str(e),row["Match_%"],None,row["All_Employees"],"Unique Fix")
                used_emps.add(e)
        df_expanded=df_expanded[~df_expanded["Employee"].isin(used_emps)]
    if not df_expanded.empty:
        emp_demand_counts=df_expanded.groupby("Employee")["Demand"].nunique().to_dict()
        df_expanded["num_factor"]=df_expanded["Employee"].map(emp_demand_counts)
        w1,w2=0.7,0.3
        df_expanded["Score"]=w1*df_expanded["Match_%"]+w2*(1/df_expanded["num_factor"])
        df_expanded=df_expanded.sort_values("Score",ascending=False)
        for _,row in df_expanded.iterrows():
            d,e=row["Demand"],row["Employee"]
            if d not in assignments and e not in used_emps:
                assignments[d]=(str(e),row["Match_%"],row["Score"],row["All_Employees"],"Score Unique")
                used_emps.add(e)
        for _,row in df_expanded.iterrows():
            d,e=row["Demand"],row["Employee"]
            if d not in assignments:
                assignments[d]=(str(e)+"*",row["Match_%"],row["Score"],row["All_Employees"],"Score Reuse")
    final=pd.DataFrame(
        [(d,emp,match,score,all_emps,method) for d,(emp,match,score,all_emps,method) in assignments.items()],
        columns=["Demand","Assigned_Employee","Match_%","Score","All_Employees","Method"]
    )
    return final

# ---- merge details
def merge_details(final_df, df_demand, df_employee):
    merged=final_df.merge(df_demand,left_on="Demand",right_on="Request profile number",how="left")
    merged=merged.merge(df_employee,left_on="Assigned_Employee",right_on="EMPID",how="left",
                        suffixes=("_demand","_employee"))
    return merged

# ---------------- STREAMLIT APP ----------------
st.title("Demand ↔ Employee Assignment Pipeline")

demand_file=st.file_uploader("Upload Demand File",type=["xls","xlsx"])
employee_file=st.file_uploader("Upload Employee File",type=["xls","xlsx"])

if demand_file and employee_file:
    df_demand=pd.read_excel(demand_file)
    df_employee=pd.read_excel(employee_file)

    # Step 1: Clean
    df_demand=clean_demand(df_demand)
    df_employee=clean_employee(df_employee)

    # Step 2: Match
    df_matches=match_demand_employee(df_demand,df_employee)

    # Step 3: First Summary
    df_summary1=demand_employee_summary(df_matches)

    # Step 4: Merge details (preview)
    df_matches_full=df_matches.merge(df_demand,on="Request profile number",how="left").merge(df_employee,on="EMPID",how="left")
    st.subheader("First Summary with Details")
    st.write(df_matches_full.head())

    # Step 5: Ask User Filters here
    st.subheader("Choose Filter Options")
    st.markdown("""
    **Available Filters:**  
    - `1` → Offshore Only  
    - `2` → Onsite Only  
    - `3` → Both Onsite/Offshore Match  
    - `4` → Band Logic  
    """)
    options = st.text_input("Enter filter options (e.g., 1,2,3,4):")
    options = [int(x.strip()) for x in options.split(",") if x.strip().isdigit()]

    if options:
        # Step 6: Apply Filters
        df_filtered=filter_data(df_matches_full, options)

        # Step 7: Second Summary
        df_summary2=demand_employee_summary(df_filtered)
        st.subheader("Second Summary (After Filter)")
        st.write(df_summary2.head())

        # Step 8: Starvation Logic
        df_starvation=starvation_logic(df_summary2)

        # Step 9: Final Merge
        df_final=merge_details(df_starvation, df_demand, df_employee)
        st.subheader("Final Starvation Assignments with Details")
        st.write(df_final.head())

        # Step 10: Download only Final Output
        df_final.to_excel("Final_Output.xlsx", index=False)
        with open("Final_Output.xlsx","rb") as f:
            st.download_button("Download Final Excel", f, "Final_Output.xlsx")
