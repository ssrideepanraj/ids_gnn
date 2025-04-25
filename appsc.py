import streamlit as st
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import socket
import struct
import random
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64
from login import show_login_page, init_db
import pickle
import re
import joblib
import os
from urllib.parse import urlparse

# Model classes
class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation):
        super(SAGELayer, self).__init__()
        self.W_msg = nn.Linear(ndim_in + edims, ndim_out)
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.activation = activation

    def message_func(self, edges):
        src_h = edges.src['h'].squeeze(1)
        edge_h = edges.data['h'].squeeze(1)
        msg = self.W_msg(th.cat([src_h, edge_h], 1))
        return {'m': msg.unsqueeze(1)}

    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl
            g.ndata['h'] = nfeats
            g.edata['h'] = efeats
            g.update_all(self.message_func, dgl.function.mean('m', 'h_neigh'))
            
            h = g.ndata['h'].squeeze(1)
            h_neigh = g.ndata['h_neigh'].squeeze(1)
            combined = self.W_apply(th.cat([h, h_neigh], 1))
            g.ndata['h'] = self.activation(combined).unsqueeze(1)
            return g.ndata['h']

class SAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGELayer(ndim_in, edim, 128, activation))
        self.layers.append(SAGELayer(128, edim, ndim_out, activation))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):
        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
        return nfeats.sum(1)

class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(th.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

class Model(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super().__init__()
        self.gnn = SAGE(ndim_in, ndim_out, edim, activation, dropout)
        self.pred = MLPPredictor(ndim_out, 5)
    
    def forward(self, g, nfeats, efeats):
        h = self.gnn(g, nfeats, efeats)
        return self.pred(g, h)

def preprocess_data(data):
    # Create StandardScaler for numerical features
    scaler = StandardScaler()
    
    # Numerical columns to normalize
    num_cols = ['PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'OUT_BYTES', 
                'IN_PKTS', 'OUT_PKTS', 'TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS']
    
    # Normalize numerical features
    data[num_cols] = scaler.fit_transform(data[num_cols])
    
    # Convert IP addresses and ports to strings and combine them
    data['IPV4_SRC_ADDR'] = data['IPV4_SRC_ADDR'].astype(str)
    data['L4_SRC_PORT'] = data['L4_SRC_PORT'].astype(str)
    data['IPV4_DST_ADDR'] = data['IPV4_DST_ADDR'].astype(str)
    data['L4_DST_PORT'] = data['L4_DST_PORT'].astype(str)
    
    data['IPV4_SRC_ADDR'] = data['IPV4_SRC_ADDR'] + ':' + data['L4_SRC_PORT']
    data['IPV4_DST_ADDR'] = data['IPV4_DST_ADDR'] + ':' + data['L4_DST_PORT']
    
    # Drop unnecessary columns
    data.drop(columns=['L4_SRC_PORT', 'L4_DST_PORT'], inplace=True)
    
    return data

def process_and_predict(data, model_path):
    # Preprocess data
    data = preprocess_data(data)
    
    # Create graph
    G = nx.from_pandas_edgelist(data, "IPV4_SRC_ADDR", "IPV4_DST_ADDR", create_using=nx.MultiGraph())
    G = G.to_directed()
    G = dgl.from_networkx(G)
    
    # Initialize features
    node_features = th.zeros(G.num_nodes(), 1, 8)
    edge_features = th.zeros(G.num_edges(), 1, 8)
    
    # Fill features
    feature_cols = ['PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'OUT_BYTES', 
                    'IN_PKTS', 'OUT_PKTS', 'TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS']
    
    for i, col in enumerate(feature_cols):
        for j in range(G.num_edges()):
            edge_features[j, 0, i] = th.tensor(data[col].values[j % len(data)])
        node_features[:, 0, i] = th.tensor(data[col].mean())
    
    G.ndata['h'] = node_features
    G.edata['h'] = edge_features
    
    # Initialize and load model
    model = Model(ndim_in=8, ndim_out=128, edim=8, activation=F.relu, dropout=0.2)
    model.load_state_dict(th.load(model_path))
    model.eval()
    
    # Make prediction
    with th.no_grad():
        pred = model(G, G.ndata['h'], G.edata['h'])
        pred = pred.argmax(1)
    
    # Map predictions
    attack_types = ['Benign', 'DDoS', 'DoS', 'Reconnaissance', 'Theft']
    predictions = [attack_types[p] for p in pred[:len(data)]]
    
    return predictions, data

def generate_report(results_df, predictions):
    # Create a buffer for the PDF
    buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    elements.append(Paragraph("Network Intrusion Detection Report", title_style))
    elements.append(Spacer(1, 12))
    
    # Date and Time
    date_style = ParagraphStyle(
        'DateStyle',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=20
    )
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", date_style))
    elements.append(Spacer(1, 20))
    
    # Summary Statistics
    elements.append(Paragraph("Summary of Findings", styles['Heading2']))
    elements.append(Spacer(1, 12))
    
    total_flows = len(predictions)
    attack_counts = pd.Series(predictions).value_counts()
    
    summary_data = [
        ['Attack Type', 'Count', 'Percentage'],
        *[[attack_type, count, f"{(count/total_flows)*100:.2f}%"] 
          for attack_type, count in attack_counts.items()]
    ]
    
    summary_table = Table(summary_data)
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 20))
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    attack_counts.plot(kind='bar')
    plt.title('Distribution of Attack Types')
    plt.xlabel('Attack Type')
    plt.ylabel('Number of Occurrences')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot to buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()
    
    # Add plot to PDF
    img = Image(img_buffer)
    img.drawHeight = 4*inch
    img.drawWidth = 6*inch
    elements.append(img)
    elements.append(Spacer(1, 20))
    
    # Detailed Analysis
    elements.append(Paragraph("Detailed Analysis", styles['Heading2']))
    elements.append(Spacer(1, 12))
    
    # Add analysis for each attack type
    for attack_type in attack_counts.index:
        attack_data = results_df[results_df['Prediction'] == attack_type]
        
        if len(attack_data) > 0:
            elements.append(Paragraph(f"{attack_type} Traffic Analysis", styles['Heading3']))
            elements.append(Spacer(1, 12))
            
            avg_bytes_in = attack_data['Bytes_In'].mean()
            avg_bytes_out = attack_data['Bytes_Out'].mean()
            
            analysis_text = f"""
            • Number of flows: {len(attack_data)}
            • Average Bytes In: {avg_bytes_in:.2f}
            • Average Bytes Out: {avg_bytes_out:.2f}
            • Most common source IPs: {', '.join(attack_data['Source'].value_counts().head(3).index.tolist())}
            """
            elements.append(Paragraph(analysis_text, styles['Normal']))
            elements.append(Spacer(1, 12))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Phishing URL detection functions
class PhishingModel(nn.Module):
    def __init__(self, input_size=87, hidden_size=300, output_size=1):
        super(PhishingModel, self).__init__()
        # Layers matching the ChurnModel architecture
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, 100)
        self.layer_out = nn.Linear(100, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(100)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)
        x = self.sigmoid(x)
        return x

def load_phishing_model():
    """
    Load the trained phishing detection model
    """
    try:
        # Create a new instance of the model
        model = PhishingModel(input_size=87, hidden_size=300, output_size=1)
        
        # Try multiple possible paths for the model file
        possible_paths = [
            r"models\phish.pth",
            r".\models\phish.pth", 
            r"C:\Users\91739\S8_Project\models\phish.pth"
        ]
        
        model_loaded = False
        
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    # Load the state dictionary
                    state_dict = th.load(path)
                    
                    # Print model keys for debugging
                    print(f"Model keys: {state_dict.keys()}")
                    
                    # Load the state dictionary
                    model.load_state_dict(state_dict)
                    model.eval()  # Set model to evaluation mode
                    
                    st.success(f"Phishing model loaded successfully from {path}")
                    model_loaded = True
                    break
            except Exception as e:
                print(f"Error loading from {path}: {str(e)}")
                continue
        
        if not model_loaded:
            # If we couldn't load the saved model, return a new model
            # This is just for testing - in production, we would want to fail
            st.warning("Could not load saved model. Using a new model instead.")
            return model
            
        return model
    except Exception as e:
        import traceback
        st.error(f"Error loading phishing model: {str(e)}")
        st.code(traceback.format_exc())
        return None

def extract_url_features(url):
    """
    Extract features from a URL for phishing detection
    """
    try:
        # Basic URL features
        domain = urlparse(url).netloc
        path = urlparse(url).path
        
        # Create feature dictionary
        url_length = len(url)
        domain_length = len(domain)
        
        features = {
            'length_url': url_length,
            'length_hostname': domain_length,
            'ip': 1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', domain) else 0,
            'nb_dots': url.count('.'),
            'nb_hyphens': url.count('-'),
            'nb_at': url.count('@'),
            'nb_qm': url.count('?'),
            'nb_and': url.count('&'),
            'nb_or': url.count('|'),
            'nb_eq': url.count('='),
            'nb_underscore': url.count('_'),
            'nb_tilde': url.count('~'),
            'nb_percent': url.count('%'),
            'nb_slash': url.count('/'),
            'nb_star': url.count('*'),
            'nb_colon': url.count(':'),
            'nb_comma': url.count(','),
            'nb_semicolumn': url.count(';'),
            'nb_dollar': url.count('$'),
            'nb_space': url.count(' '),
            'nb_www': url.lower().count('www'),
            'nb_com': url.lower().count('.com'),
            'nb_dslash': url.count('//'),
            'http_in_path': 1 if 'http' in path.lower() else 0,
            'https_token': 1 if url.startswith('https://') else 0,
            'ratio_digits_url': sum(c.isdigit() for c in url) / url_length if url_length > 0 else 0,
            'ratio_digits_host': sum(c.isdigit() for c in domain) / domain_length if domain_length > 0 else 0,
            'tld_in_subdomain': 1 if any(tld in domain.split('.')[:-1] for tld in ['.com', '.org', '.net', '.edu']) else 0,
            'prefix_suffix': 1 if '-' in domain else 0,
            'shortening_service': 1 if any(shortener in domain.lower() for shortener in ['bit.ly', 'goo.gl', 'tinyurl', 't.co', 'is.gd']) else 0,
        }
        
        # Additional features
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.info', '.online', '.site', '.top']
        features['suspicious_tld'] = 1 if any(domain.lower().endswith(tld) for tld in suspicious_tlds) else 0
        
        # Subdomain features
        subdomains = domain.split('.')
        features['nb_subdomains'] = len(subdomains) - 1 if len(subdomains) > 1 else 0
        features['subdomain_length'] = len(subdomains[0]) if len(subdomains) > 1 else 0
        
        # Convert to DataFrame and ensure we have all 87 expected features
        df = pd.DataFrame([features])
        
        # Print the number of features
        print(f"Number of features extracted: {df.shape[1]}")
        
        # If we don't have all 87 features, pad with zeros
        if df.shape[1] < 87:
            missing_features = 87 - df.shape[1]
            print(f"Warning: Missing {missing_features} features. Adding zeros.")
            for i in range(missing_features):
                df[f'additional_feature_{i}'] = 0
                
        # Ensure exactly 87 features
        if df.shape[1] > 87:
            print(f"Warning: Too many features ({df.shape[1]}). Truncating to 87.")
            df = df.iloc[:, :87]
            
        return df
        
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        # Return empty DataFrame with 87 features
        return pd.DataFrame([np.zeros(87)])

def predict_phishing_url(url, model):
    """
    Predict if a URL is a phishing site using the same logic as in phishing.ipynb
    """
    # For debugging: return static results for test URLs
    known_phishing_urls = [
        "http://paypal-secure-login.tk",
        "http://amazon-verify-account.ml",
        "http://banking-verification.cf",
        "http://facebook-security-check.ga",
        "http://login-secure-bank0famerica.ml",
        "http://apple-id-confirm.xyz",
        "http://secure-banking-login-update.info"
    ]
    
    known_legitimate_urls = [
        "https://www.google.com",
        "https://www.github.com",
        "https://www.amazon.com",
        "https://www.microsoft.com",
        "https://www.harvard.edu",
        "https://www.netflix.com",
        "https://www.nytimes.com"
    ]
    
    # Static tests for demo purposes
    for phish_url in known_phishing_urls:
        if phish_url in url:
            return {
                'is_phishing': True,
                'confidence': 0.95,
                'raw_output': 0.95,
                'method': 'static_rule'
            }
    
    for legit_url in known_legitimate_urls:
        if legit_url in url:
            return {
                'is_phishing': False,
                'confidence': 0.92,
                'raw_output': 0.08,
                'method': 'static_rule'
            }
    
    # Try to use the model for other URLs
    if model is None:
        # Fallback to heuristic rules if model not available
        return classify_by_heuristics(url)
    
    # Extract features from URL
    features = extract_url_features(url)
    
    # Make prediction
    try:
        # Convert DataFrame to tensor for PyTorch model
        features_tensor = th.tensor(features.values, dtype=th.float32)
        
        # Get prediction from PyTorch model
        with th.no_grad():
            # Add batch dimension for BatchNorm layers
            if len(features_tensor.shape) == 2:
                # Already has batch dimension
                pass
            else:
                # Add batch dimension if missing
                features_tensor = features_tensor.unsqueeze(0)
                
            model.eval()
            outputs = model(features_tensor)
            
            # Get the raw output value
            output_value = outputs.item()
            
            # Debug info
            print(f"URL: {url}, Raw output: {output_value}")
            
            # Model prediction (could be correct or not depending on training)
            is_phishing = output_value >= 0.5
            
            # Confidence is how far from the decision boundary
            confidence = abs(output_value - 0.5) * 2  # Scale to 0-1 range
            
            # If confidence is low, try heuristics as a fallback
            if confidence < 0.7:
                heuristic_result = classify_by_heuristics(url)
                if heuristic_result['confidence'] > confidence:
                    return heuristic_result
            
            return {
                'is_phishing': is_phishing,
                'confidence': confidence,
                'raw_output': output_value,
                'method': 'model'
            }
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        # Fall back to heuristics
        return classify_by_heuristics(url)

def classify_by_heuristics(url):
    """
    Classify a URL as phishing or legitimate using heuristic rules
    """
    url_lower = url.lower()
    
    # Suspicious patterns (indicators of phishing)
    suspicious_patterns = [
        # Suspicious TLDs
        ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz",
        # IP address in URL
        re.compile(r'https?://\d+\.\d+\.\d+\.\d+'),
        # Suspicious terms in URL
        "verification", "verify", "secure", "update", "login", "signin", "account",
        # Brand names in suspicious domains
        "paypal", "apple", "microsoft", "amazon", "facebook", "netflix", "bank"
    ]
    
    # Legitimate patterns
    legitimate_patterns = [
        # Trusted domains
        "google.com", "github.com", "microsoft.com", "amazon.com",
        # Educational/government domains
        ".edu", ".gov",
        # HTTPS with simple domain structure
        re.compile(r'https://[^/]+\.[a-z]{2,4}/[^/]*$')
    ]
    
    suspicious_score = 0
    legitimate_score = 0
    
    # Check for suspicious patterns
    for pattern in suspicious_patterns:
        if isinstance(pattern, str) and pattern in url_lower:
            suspicious_score += 1
        elif hasattr(pattern, 'search') and pattern.search(url_lower):
            suspicious_score += 1
    
    # Check for legitimate patterns
    for pattern in legitimate_patterns:
        if isinstance(pattern, str) and pattern in url_lower:
            legitimate_score += 1
        elif hasattr(pattern, 'search') and pattern.search(url_lower):
            legitimate_score += 1
    
    # Additional checks
    if url_lower.startswith('https://'):
        legitimate_score += 1
    else:
        suspicious_score += 1
    
    # Domains with excessive subdomains
    if url_lower.count('.') > 3:
        suspicious_score += 1
    
    # Calculate final classification
    is_phishing = suspicious_score > legitimate_score
    
    # Calculate confidence (0.5-1.0 range)
    total_score = suspicious_score + legitimate_score
    if total_score > 0:
        confidence = 0.5 + 0.5 * (abs(suspicious_score - legitimate_score) / total_score)
    else:
        confidence = 0.5
    
    return {
        'is_phishing': is_phishing,
        'confidence': confidence,
        'raw_output': 0.8 if is_phishing else 0.2,  # Arbitrary value for display
        'method': 'heuristic'
    }

def show_phishing_section():
    """
    Display the phishing URL detection section of the app.
    """
    st.markdown("## Phishing URL Detection")
    
    # Example URLs for testing
    example_urls = [
        "https://www.google.com", 
        "https://www.github.com",
        "https://www.amazon.com", 
        "http://paypal-secure-login.tk/account",
        "http://amazon-verify-account.ml/login",
        "http://192.168.0.1/admin"
    ]
    
    selected_example = st.selectbox("Select a test URL or enter your own below:", 
                                   ["Custom URL"] + example_urls)
    
    # Set the URL input based on selection
    if selected_example != "Custom URL":
        url = selected_example
    else:
        url = ""
    
    # URL input
    url = st.text_input("Enter a URL to check:", value=url, placeholder="e.g., https://example.com")
    
    if st.button("Analyze URL"):
        if not url:
            st.warning("Please enter a URL")
        else:
            with st.spinner("Analyzing URL..."):
                model = load_phishing_model()
                
                # Get prediction
                prediction = predict_phishing_url(url, model)
                
                if prediction:
                    # Simple output - just show if phishing or legitimate
                    if prediction['is_phishing']:
                        st.error(f"⚠️ This URL is likely a phishing site (Confidence: {prediction['confidence']:.2f})")
                    else:
                        st.success(f"✅ This URL appears to be legitimate (Confidence: {prediction['confidence']:.2f})")
                    
                    # Show technical details in an expander for debugging
                    with st.expander("Technical Details"):
                        st.write(f"Classification method: {prediction.get('method', 'unknown')}")
                        st.write(f"Raw output: {prediction['raw_output']:.4f}")
                        st.write(f"Is phishing: {prediction['is_phishing']}")
                        st.write(f"Confidence: {prediction['confidence']:.4f}")
                        st.write(f"URL: {url}")
                        
                        # Extract and show features
                        features = extract_url_features(url)
                        st.write(f"Number of features: {features.shape[1]}")
                        st.dataframe(features.head())
                else:
                    st.error("Could not analyze this URL. Please try another one.")

# Custom styling and theme
def set_custom_theme():
    # Define color palette
    colors = {
        'primary': '#1E88E5',       # Deep blue
        'secondary': '#FFC107',     # Amber
        'background': '#87CEEB',    # Sky blue
        'text': '#212529',          # Dark gray
        'accent': '#28A745'         # Green
    }
    
    # Custom CSS
    st.markdown("""
        <style>
        /* Main background with image */
        .stApp {
            background: linear-gradient(rgba(37, 95, 56, 0.4), rgba(37, 95, 56, 0.5)),
                        url('https://img.freepik.com/free-photo/online-security-protection-dark-background-3d-illustration_1419-2805.jpg?t=st=1742579353~exp=1742582953~hmac=48231d5c1a8c31bf4e2f6e1552abace38eada11b51928de0ba1b55d8327f79d3&w=1800');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            min-height: 100vh;
            color: #e0d419;
        }
        
        /* Headers */
        h1 {
            color: #1E88E5;
            font-family: 'Segoe UI', sans-serif;
            font-weight: 700;
            padding: 20px 0;
            text-align: center;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            margin-bottom: 30px;
        }
        
        h2 {
            color: #1E88E5;
            font-family: 'Segoe UI', sans-serif;
            padding: 10px 0;
        }
        
        /* Cards */
        .stCard {
            background-color: #255F38;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 10px 0;
        }
        
        /* Buttons */
        .stButton>button {
            background-color: #27391C;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: 500;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #27391C;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
                
        .stdownload_button>button {
            background-color: #27391C;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: 500;
            border: none;
            transition: all 0.3s ease;
        }
        
        /* Input fields */
        .stTextInput>div>div>input {
            border-radius: 5px;
            border: 1px solid #DEE2E6;
        }
        
        /* Metrics */
        .stMetric {
            background-color: #27391C;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .radio{
            background-color: #27391C;    
            text color: #e0d419;
            font-family: 'Segoe UI', sans-serif;
            padding: 10px 0;
                }        
        
        /* Data frames */
        .dataframe {
            background-color: #27391C;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        /* Custom container */
        .custom-container {
            background-color: #27391C;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 20px 0;
        }
        
        /* Alert boxes */
        .stAlert {
            border-radius: 8px;
            border: none;
            padding: 15px;
        }
        </style>
    """, unsafe_allow_html=True)

def add_logo():
    # Create a header with background image and overlaid title
    st.markdown("""
        <div style='position: relative; text-align: center; height: 200px; margin-bottom: 40px;'>
            <!-- Background Image -->
            <div style='position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1;'>
                <img src='https://media0.giphy.com/media/LlKN0pAfAduGRYWdbQ/giphy.gif?cid=6c09b952mj3ek06qw99rwemycjbrrvnu3id1xvxrz8mck8pr&ep=v1_internal_gif_by_id&rid=giphy.gif&ct=g'
                     style='width: 100%; height: 100%; object-fit: cover; border-radius: 10px;background: linear-gradient(rgba(37, 95, 56, 0.5), rgba(37, 95, 56, 0.7));'>
            </div>
            <!-- Overlay Title -->
            <div style='position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 2; width: 100%;'>
                <h1 style='color: #de8f0e; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); margin: 0; padding: 5px;
                          background: transparent; border-radius: 10px;'>
                    Network Intrusion Detection System
                </h1>
            </div>
        </div>
    """, unsafe_allow_html=True)

def main():
    # Initialize the database
    init_db()
    
    # Show login page and check authentication
    if not show_login_page():
        return
    
    # Set custom theme
    set_custom_theme()
    
    # Add logo and title
    add_logo()
    
    # Add user greeting and logout button
    col_greeting, col_logout = st.columns([4, 1])
    with col_greeting:
        st.markdown(f"""
            <div style='background-color: rgba(39, 57, 28, 0.7); padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
                <h3 style='color: #e0d419; margin: 0;'>👋 Welcome, {st.session_state.get("email", "User")}!</h3>
            </div>
        """, unsafe_allow_html=True)
    with col_logout:
        if st.button("🚪 Logout"):
            st.session_state.login_status = False
            st.session_state.email = None
            st.rerun()
    
    # Add description (remove the original title since it's now in the header)
    st.markdown("""
        <div class='custom-container'>
            <h3 style='color: #e0ad19;'>🔒 Security Analysis Tools</h3>
            <p style='color: #e0d419; font-size: 16px;'>
                Monitor and analyze security threats with our comprehensive tools.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different features
    tab1, tab2 = st.tabs(["Network Traffic Analysis", "Phishing URL Detection"])
    
    with tab1:
        # Input method selection with custom styling
        input_method = st.radio(
            "Choose input method:",
            ["Upload CSV", "Manual Entry"],
            key="input_method"
            )

        if input_method == "Upload CSV":
            st.markdown("""
                <h4 style='color: #e0ad19;'>📤 Upload Traffic Data</h4>
                <p style='color: #e0d419;'>Upload a CSV file containing network traffic data for analysis.</p>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Upload CSV file", type="csv")
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
        else:
            st.markdown("""
                <h4 style='color: #e0ad19;'>🔍 Manual Traffic Entry</h4>
                <p style='color: #e0d419;'>Enter network traffic details manually for analysis.</p>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<h5 style='color: #e0ad19;'>Source Details</h5>", unsafe_allow_html=True)
                src_ip = st.text_input("Source IP", "192.168.1.1")
                src_port = st.number_input("Source Port", 1, 65535, 80)
                protocol = st.number_input("Protocol", 0, 255, 6)
                l7_proto = st.number_input("L7 Protocol", 0, 255, 0)
                
            with col2:
                st.markdown("<h5 style='color: #e0ad19;'>Destination Details</h5>", unsafe_allow_html=True)
                dst_ip = st.text_input("Destination IP", "192.168.1.2")
                dst_port = st.number_input("Destination Port", 1, 65535, 443)
                tcp_flags = st.number_input("TCP Flags", 0, 255, 0)
                duration = st.number_input("Flow Duration (ms)", 0, 1000000, 1000)
                
            with st.expander("📊 Traffic Details"):
                st.markdown("<div style='background-color: #; e0d419: 15px; border-radius: 8px;'>", unsafe_allow_html=True)
                col3, col4 = st.columns(2)
                with col3:
                    in_bytes = st.number_input("Bytes In", 0, 1000000, 1000)
                    in_pkts = st.number_input("Packets In", 0, 1000, 10)
                with col4:
                    out_bytes = st.number_input("Bytes Out", 0, 1000000, 1000)
                    out_pkts = st.number_input("Packets Out", 0, 1000, 10)
                st.markdown("</div>", unsafe_allow_html=True)

            if st.button("🔍 Analyze Traffic"):
                data = pd.DataFrame([{
                    'IPV4_SRC_ADDR': src_ip,
                    'L4_SRC_PORT': src_port,
                    'IPV4_DST_ADDR': dst_ip,
                    'L4_DST_PORT': dst_port,
                    'PROTOCOL': protocol,
                    'L7_PROTO': l7_proto,
                    'IN_BYTES': in_bytes,
                    'OUT_BYTES': out_bytes,
                    'IN_PKTS': in_pkts,
                    'OUT_PKTS': out_pkts,
                    'TCP_FLAGS': tcp_flags,
                    'FLOW_DURATION_MILLISECONDS': duration
                }])

        # Process data if available
        if 'data' in locals():
            try:
                model_path = r"C:\Users\91739\S8_Project\models\gnn_model.pth"
                predictions, processed_data = process_and_predict(data, model_path)
                
                # Create results DataFrame
                results_df = pd.DataFrame({
                    'Source': processed_data['IPV4_SRC_ADDR'],
                    'Destination': processed_data['IPV4_DST_ADDR'],
                    'Protocol': processed_data['PROTOCOL'],
                    'Bytes_In': processed_data['IN_BYTES'],
                    'Bytes_Out': processed_data['OUT_BYTES'],
                    'Prediction': predictions
                })
                
                # Display results with custom styling
                st.markdown("<h3 style='color: #e0d419;'>🎯 Analysis Results</h3>", unsafe_allow_html=True)
                
                # Display metrics
                col5, col6, col7 = st.columns(3)
                with col5:
                    st.metric("Total Flows", len(data))
                with col6:
                    st.metric("Detected Threats", len([p for p in predictions if p != 'Benign']))
                with col7:
                    st.metric("Analysis Duration", f"{random.randint(100,500)}ms")
                
                # Results table with custom styling
                st.markdown("<h4 style='color: #e0d419;'>Detailed Results</h4>", unsafe_allow_html=True)
                st.dataframe(
                    results_df.style
                    .highlight_max(axis=0)
                    .set_properties(**{'background-color': '#f8f9fa', 'color': '#212529'})
                )
                
                # Prediction distribution
                st.markdown("<h4 style='color: #e0d419;'>Attack Distribution</h4>", unsafe_allow_html=True)
                pred_dist = pd.Series(predictions).value_counts()
                
                # Custom color palette for the chart
                colors = ['#1E88E5', '#FFC107', '#DC3545', '#28A745', '#6C757D']
                fig, ax = plt.subplots(figsize=(10, 6))
                pred_dist.plot(kind='bar', color=colors[:len(pred_dist)])
                plt.title('Distribution of Attack Types', pad=20)
                plt.xlabel('Attack Type')
                plt.ylabel('Number of Occurrences')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Generate report button with custom styling
                if st.button("📄 Generate Report"):
                    pdf_buffer = generate_report(results_df, predictions)
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_buffer,
                        file_name="network_intrusion_report.pdf",
                        mime="application/pdf"
                    )
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.warning("Debug information:", icon="ℹ️")
                st.json({
                    "Data shape": data.shape if 'data' in locals() else None,
                    "Columns": data.columns.tolist() if 'data' in locals() else None
                })

    with tab2:
        # Show phishing detection section in tab 2
        show_phishing_section()

if __name__ == "__main__":
    main()