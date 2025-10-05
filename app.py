# ============================
# Hybrid Network Intrusion Analysis Dashboard
# Random Forest (Classification) + Apriori (Unsupervised)
# ============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import networkx as nx
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
# -----------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------
st.set_page_config(page_title="Hybrid Cyber Model", layout="wide")
st.title("🛡 Hybrid Network Intrusion Analysis Dashboard")
st.markdown("Explore your network traffic using **Random Forest** for attack classification "
            "and **Apriori** for association rule discovery.")

# Sidebar
st.sidebar.title("Navigation")

mode = st.sidebar.radio("Select Mode:", ["Apriori (Unsupervised)", "Random Forest (Classification)"])
 # ---- Feature Information Section ----
with st.expander("📘 View Feature Descriptions (UNSW-NB15 Columns)"):
        feature_info = {
            'srcip': 'Source IP address',
            'sport': 'Source port number',
            'dstip': 'Destination IP address',
            'dsport': 'Destination port number',
            'proto': 'Transaction protocol (e.g. TCP, UDP)',
            'state': 'Connection state (ACC, FIN, etc.)',
            'dur': 'Record total duration',
            'sbytes': 'Source to destination bytes',
            'dbytes': 'Destination to source bytes',
            'sttl': 'Source TTL (time-to-live)',
            'dttl': 'Destination TTL',
            'sloss': 'Source packets retransmitted or dropped',
            'dloss': 'Destination packets retransmitted or dropped',
            'service': 'Service type (http, dns, etc.)',
            'Sload': 'Source bits per second',
            'Dload': 'Destination bits per second',
            'Spkts': 'Packets from source to destination',
            'Dpkts': 'Packets from destination to source',
            'swin': 'Source TCP window advertisement value',
            'dwin': 'Destination TCP window advertisement value',
            'stcpb': 'Source TCP base sequence number',
            'dtcpb': 'Destination TCP base sequence number',
            'smeansz': 'Mean of packet size transmitted by source',
            'dmeansz': 'Mean of packet size transmitted by destination',
            'trans_depth': 'HTTP request/response pipeline depth',
            'res_bdy_len': 'Uncompressed response body length',
            'Sjit': 'Source jitter (ms)',
            'Djit': 'Destination jitter (ms)',
            'Stime': 'Record start timestamp',
            'Ltime': 'Record last timestamp',
            'Sintpkt': 'Source interpacket arrival time (ms)',
            'Dintpkt': 'Destination interpacket arrival time (ms)',
            'tcprtt': 'TCP connection setup round-trip time',
            'synack': 'Time between SYN and SYN-ACK packets',
            'ackdat': 'Time between SYN-ACK and ACK packets',
            'is_sm_ips_ports': '1 if src/dst IP and ports are same, else 0',
            'ct_state_ttl': 'Count of state-ttl pairs',
            'ct_flw_http_mthd': 'Count of HTTP methods (GET, POST)',
            'is_ftp_login': '1 if FTP login succeeded, else 0',
            'ct_ftp_cmd': 'Count of FTP commands in session',
            'ct_srv_src': 'Connections with same service & source IP (last 100)',
            'ct_srv_dst': 'Connections with same service & destination IP (last 100)',
            'ct_dst_ltm': 'Connections to same destination IP (last 100)',
            'ct_src_ ltm': 'Connections from same source IP (last 100)',
            'ct_src_dport_ltm': 'Connections with same src & dst port (last 100)',
            'ct_dst_sport_ltm': 'Connections with same dst & src port (last 100)',
            'ct_dst_src_ltm': 'Connections with same src & dst IP (last 100)',
            'attack_cat': 'Attack category (e.g. DoS, Fuzzers, etc.)',
            'label': '0 = normal, 1 = attack'
        }

        selected_feature = st.selectbox("Select a feature to view its meaning:", list(feature_info.keys()))
        st.info(f"**{selected_feature}:** {feature_info[selected_feature]}")
# =========================
# APRIORI MODE (Final Clean + Binning Table + Simplified Rules)
# =========================
if mode == "Apriori (Unsupervised)":
    st.header("🧩 Apriori Rule Mining")
    st.write("Discover frequent itemsets and association rules in your network data. "
             "Labels (attack/normal) are not used for training — only for interpretation.")

    # ---- Upload Dataset ----
    uploaded = st.file_uploader("📂 Upload your dataset (CSV)", type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("✅ Loaded dataset:", df.shape)

        # Drop irrelevant columns
        drop_cols = [c for c in ['srcip', 'dstip', 'sport', 'dsport', 'attack_cat', 'label'] if c in df.columns]
        df_ap = df.drop(columns=drop_cols, errors='ignore').copy()

        # Identify categorical and numeric columns
        cat_cols = df_ap.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = df_ap.select_dtypes(include=[np.number]).columns.tolist()

        # Sidebar controls
        st.sidebar.subheader("Apriori Parameters")
        selected_cat = st.sidebar.multiselect("Categorical features:", cat_cols, default=cat_cols[:2])
        selected_num = st.sidebar.multiselect("Numeric features (will be binned):", num_cols, default=num_cols[:2])
        min_support = st.sidebar.slider("Min Support", 0.001, 0.5, 0.01, 0.001)
        min_conf = st.sidebar.slider("Min Confidence", 0.1, 1.0, 0.5, 0.05)
        max_len = st.sidebar.slider("Max Rule Length", 1, 5, 3)
        bins = st.sidebar.slider("Numeric bins", 2, 8, 4)

        if st.button("Run Apriori"):
            df_ap = df_ap.copy()
            bin_summary = []

            # ----- Numeric Binning -----
            for c in selected_num:
                if c in df_ap.columns:
                    df_ap[c] = pd.to_numeric(df_ap[c], errors='coerce')
                    valid_idx = df_ap[c].notnull()
                    if valid_idx.sum() > 0:
                        kbd = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
                        binned = kbd.fit_transform(df_ap.loc[valid_idx, [c]]).astype(int).flatten()
                        df_ap.loc[valid_idx, f"{c}_bin"] = binned.astype(str)

                        # Store bin edges for summary table
                        bin_edges = np.percentile(df_ap.loc[valid_idx, c], np.linspace(0, 100, bins + 1))
                        for i in range(bins):
                            bin_summary.append({
                                "Feature": c,
                                "Bin": i,
                                "Range Start": round(bin_edges[i], 3),
                                "Range End": round(bin_edges[i + 1], 3)
                            })
                    else:
                        df_ap[f"{c}_bin"] = "missing"

            # ---- Display binning summary ----
            if bin_summary:
                st.subheader("📊 Numeric Feature Binning Overview")
                st.dataframe(pd.DataFrame(bin_summary))

            # ---- One-hot encode categorical + binned numeric ----
            onehot_cols = selected_cat + [f"{c}_bin" for c in selected_num if f"{c}_bin" in df_ap.columns]
            df_onehot = pd.get_dummies(df_ap[onehot_cols].astype(str), prefix_sep='=').astype(bool).fillna(False)

            st.write("✅ One-hot encoded shape:", df_onehot.shape)

            # Optional sampling for performance
            if df_onehot.shape[0] > 5000:
                df_sample = df_onehot.sample(n=5000, random_state=42)
                st.info("⚙️ Running Apriori on 5000-row sample to improve speed.")
            else:
                df_sample = df_onehot

            # ---- Run Apriori ----
            frequent_itemsets = apriori(df_sample, min_support=min_support, use_colnames=True, max_len=max_len)
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)

            if not rules.empty:
                rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

            st.success(f"✅ Found {len(frequent_itemsets)} itemsets and {len(rules)} rules")

            # ---- Display Frequent Itemsets ----
            st.subheader("📈 Frequent Itemsets (Top 20)")
            st.dataframe(frequent_itemsets.sort_values('support', ascending=False).head(20))

            # ---- Bar Chart ----
            if not frequent_itemsets.empty:
                fig = px.bar(
                    frequent_itemsets.sort_values('support', ascending=False).head(15),
                    x='support', y=frequent_itemsets['itemsets'].astype(str).head(15),
                    orientation='h', title="Top Frequent Itemsets"
                )
                st.plotly_chart(fig, use_container_width=True)

            # ---- Simplified Rules Table (first 5 key columns only) ----
            if not rules.empty:
                st.subheader("📜 Association Rules (Simplified)")
                rules_display = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(
                    'lift', ascending=False
                ).reset_index(drop=True)
                st.dataframe(rules_display.head(20))

                # ---- Scatter Plot ----
                fig2 = px.scatter(
                    rules, x='support', y='confidence', size='lift',
                    hover_data=['antecedents', 'consequents'],
                    title="Rule Confidence vs Support"
                )
                st.plotly_chart(fig2, use_container_width=True)

                # ---- Rule Network ----
                st.subheader("🌐 Rule Network Graph")
                top_rules = rules_display.head(30)
                G = nx.DiGraph()
                for _, row in top_rules.iterrows():
                    ants = row['antecedents'].split(', ')
                    cons = row['consequents'].split(', ')
                    for a in ants:
                        for c in cons:
                            G.add_edge(str(a), str(c), weight=row['confidence'])
                pos = nx.spring_layout(G, k=0.6, seed=42)
                plt.figure(figsize=(10, 7))
                nx.draw(G, pos, with_labels=True, node_size=800, node_color='skyblue', font_size=8)
                st.pyplot(plt.gcf())




# =====================================================
# RANDOM FOREST (CLASSIFICATION) MODE — REVISED
# =====================================================
elif mode == "Random Forest (Classification)":
    st.header("🧠 Attack Classification using Random Forest")

    uploaded = st.file_uploader("Upload your dataset for prediction (CSV)", type=["csv"])
    if uploaded:
        # Define the 49 feature column names in order
        feature_cols = [
            "srcip","sport","dstip","dsport","proto","state","dur","sbytes","dbytes","sttl","dttl",
            "sloss","dloss","service","Sload","Dload","Spkts","Dpkts","swin","dwin","stcpb","dtcpb",
            "smeansz","dmeansz","trans_depth","res_bdy_len","Sjit","Djit","Stime","Ltime","Sintpkt",
            "Dintpkt","tcprtt","synack","ackdat","is_sm_ips_ports","ct_state_ttl","ct_flw_http_mthd",
            "is_ftp_login","ct_ftp_cmd","ct_srv_src","ct_srv_dst","ct_dst_ltm","ct_src_ltm",
            "ct_src_dport_ltm","ct_dst_sport_ltm","ct_dst_src_ltm","attack_cat","Label"
        ]

        # Load CSV without headers
        df = pd.read_csv(uploaded, header=None)
        if df.shape[1] == len(feature_cols):
            df.columns = feature_cols
            st.success("✅ Column headers assigned automatically.")
        else:
            st.error(f"⚠️ Expected {len(feature_cols)} columns, but got {df.shape[1]}.")
            st.stop()

        st.write("Dataset shape:", df.shape)

        # Drop unnecessary columns for prediction
        drop_cols = ["srcip","dstip","sport","dsport","attack_cat","Label"]
        X_new = df.drop(columns=drop_cols, errors="ignore")

        # Encode categorical columns
        cat_cols = X_new.select_dtypes(include=["object","category"]).columns.tolist()
        for col in cat_cols:
            le = LabelEncoder()
            X_new[col] = le.fit_transform(X_new[col].astype(str))

        # Align with model features
        model = joblib.load("rf_unsw_model.pkl")
        model_features = model.feature_names_in_
        for col in model_features:
            if col not in X_new.columns:
                X_new[col] = 0
        X_new = X_new[model_features]

        # --- Prediction ---
        preds = model.predict(X_new)
        probs = model.predict_proba(X_new)[:, 1]
        df["Predicted_Label"] = preds
        df["Attack_Probability"] = probs

        st.subheader("🔍 Classification Results (Sample View)")
        st.dataframe(df.head())

        # --- Feature Importance ---
        fi = pd.DataFrame({
            "Feature": model_features,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(8,5))
        sns.barplot(x="Importance", y="Feature", data=fi, ax=ax)
        ax.set_title("Top 15 Important Features")
        st.pyplot(fig)

        # --- Attack vs Normal Plot ---
        fig2, ax2 = plt.subplots()
        sns.countplot(x="Predicted_Label", data=df, ax=ax2)
        ax2.set_title("Predicted Attack vs Normal Distribution")
        st.pyplot(fig2)

        # --- Evaluation Metrics (only if Label column exists) ---
        if "Label" in df.columns:
            y_true = df["Label"].astype(int)
            y_preds = preds
            y_probs = probs

            st.subheader("📊 Evaluation Metrics")
            st.write(f"**Accuracy:** {accuracy_score(y_true, y_preds):.4f}")
            st.write(f"**Precision:** {precision_score(y_true, y_preds):.4f}")
            st.write(f"**Recall:** {recall_score(y_true, y_preds):.4f}")
            st.write(f"**F1-score:** {f1_score(y_true, y_preds):.4f}")

            # Confusion Matrix
            cm = confusion_matrix(y_true, y_preds)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            ax_cm.set_title("Confusion Matrix")
            st.pyplot(fig_cm)

            # ROC-AUC (binary)
            if len(np.unique(y_true)) == 2:
                auc = roc_auc_score(y_true, y_probs)
                st.write(f"**ROC-AUC Score:** {auc:.4f}")
        else:
            st.warning("⚠️ 'Label' column not found — metrics cannot be evaluated.")

        # --- Download Results ---
        st.download_button(
            "⬇️ Download Predictions CSV",
            data=df.to_csv(index=False).encode(),
            file_name="classified_output.csv",
            mime="text/csv"
        )
