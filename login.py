import streamlit as st
import sqlite3
import hashlib
from pathlib import Path

def init_db():
    # Create a database connection
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Create table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (email TEXT PRIMARY KEY, password TEXT)''')
    conn.commit()
    conn.close()

    # Add after init_db()
    create_user("admin", "password")

def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def create_user(email, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    hashed_pw = hash_password(password)
    try:
        c.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, hashed_pw))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(email, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    hashed_pw = hash_password(password)
    c.execute("SELECT * FROM users WHERE email=? AND password=?", (email, hashed_pw))
    result = c.fetchone()
    conn.close()
    return result is not None

def show_login_page():
    st.markdown("""
        <style>
        /* Main background with image */
        .stApp {
            background: linear-gradient(rgba(37, 95, 56, 0.1), rgba(37, 95, 56, 0.3)),
                        url('https://img.freepik.com/free-photo/online-security-dark-background-3d-illustration_1419-2804.jpg?t=st=1742579122~exp=1742582722~hmac=c5f280f5ce1fc1b5fb10efee9887aa5457defde4de8ec21c66e846be497167aa&w=1800');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            min-height: 100vh;
            color: #e0d419;
        }
        
        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        .css-1d391kg {
            padding: 1rem;
        }
        
        .login-container {
            max-width: 1000px;
            margin: 50px auto;
            padding: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            border: 1px solid rgba(224, 212, 25, 0.2);
            animation: glow 3s infinite alternate;
        }
        
        @keyframes glow {
            from {
                box-shadow: 0 0 20px rgba(224, 212, 25, 0.2);
            }
            to {
                box-shadow: 0 0 30px rgba(224, 212, 25, 0.4);
            }
        }
        
        .logo-section {
            flex: 1;
            padding: 40px;
            text-align: center;
            border-right: 2px solid rgba(224, 212, 25, 0.2);
            position: relative;
        }
        
        .logo-section::after {
            content: '';
            position: absolute;
            top: 20px;
            right: -1px;
            width: 2px;
            height: calc(100% - 40px);
            background: linear-gradient(
                to bottom,
                rgba(224, 212, 25, 0),
                rgba(224, 212, 25, 0.4),
                rgba(224, 212, 25, 0)
            );
        }
        
        .login-section {
            flex: 1;
            background: rgba(39, 57, 28, 0.4);
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            backdrop-filter: blur(8px);
            margin: 20px;
            border: 1px solid rgba(224, 212, 25, 0.1);
            position: relative;
            overflow: hidden;
        }
        
        .login-section::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(
                circle at center,
                rgba(224, 212, 25, 0.1) 0%,
                rgba(224, 212, 25, 0) 50%
            );
            animation: rotate 10s linear infinite;
        }
        
        @keyframes rotate {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
        
        .login-title {
            color: #e0d419;
            font-size: 28px;
            margin-bottom: 30px;
            text-align: center;
            text-transform: uppercase;
            letter-spacing: 2px;
            font-weight: bold;
        }
        
        /* Input fields */
        .stTextInput > div > div > input {
            background-color: rgba(39, 57, 28, 0.3);
            color: #e0d419;
            border: 2px solid rgba(224, 212, 25, 0.3);
            border-radius: 8px;
            padding: 15px;
            font-size: 16px;
            transition: all 0.3s ease;
            backdrop-filter: blur(4px);
        }
        
        .stTextInput > div > div > input:focus {
            box-shadow: 0 0 15px rgba(224, 212, 25, 0.3);
            border-color: #e0d419;
            background-color: rgba(39, 57, 28, 0.5);
        }
        
        .stTextInput > div > div > input::placeholder {
            color: rgba(224, 212, 25, 0.6);
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(45deg, #e0d419, #ffd700);
            color: #255F38;
            font-weight: bold;
            width: 100%;
            padding: 12px 0;
            border-radius: 8px;
            border: none;
            margin-top: 25px;
            font-size: 18px;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(224, 212, 25, 0.4);
            background: linear-gradient(45deg, #ffd700, #e0d419);
        }
        
        .stButton > button::after {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(
                circle at center,
                rgba(255, 255, 255, 0.2) 0%,
                rgba(255, 255, 255, 0) 50%
            );
            transform: rotate(45deg);
            animation: shine 3s infinite;
        }
        
        @keyframes shine {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
        
        .link-text {
            color: #e0d419;
            text-align: center;
            margin-top: 25px;
            font-size: 14px;
            opacity: 0.8;
        }
        
        .link-text a {
            color: #e0d419;
            text-decoration: none;
            transition: all 0.3s ease;
        }
        
        .link-text a:hover {
            opacity: 1;
            text-decoration: underline;
        }
        
        .circuit-logo {
            width: 200px;
            height: 200px;
            margin-bottom: 30px;
            filter: drop-shadow(0 0 10px rgba(224, 212, 25, 0.3));
        }
        
        /* Error message styling */
        .stAlert {
            background-color: rgba(255, 0, 0, 0.1);
            color: #ff4444;
            border: 1px solid #ff4444;
            border-radius: 8px;
            padding: 10px;
            margin-top: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'login_status' not in st.session_state:
        st.session_state.login_status = False

    if not st.session_state.login_status:
        st.markdown("""
                    <h1 style="color: #e0d419; text-align: center; font-size: 40px;">
                        INTRUSION DETECTION SYSTEM
                    </h1>
                

        """, unsafe_allow_html=True)

        email = st.text_input("Email", key="email", placeholder="Enter your email")
        password = st.text_input("Password", type="password", key="password", placeholder="Enter your password")

        if st.button("Login", key="login_button"):
            if login_user(email, password):
                st.session_state.login_status = True
                st.rerun()
            else:
                st.error("Invalid email or password")

        st.markdown("""
                    </div>
                    <div class="link-text">
                        <a href="#" style="color: #e0d419;">Create an account</a>
                        &nbsp;&nbsp;|&nbsp;&nbsp;
                        <a href="#" style="color: #e0d419;">Forgot Password</a>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    return st.session_state.login_status

if __name__ == "__main__":
    init_db()
    show_login_page() 