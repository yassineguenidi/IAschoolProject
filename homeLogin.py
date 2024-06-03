import streamlit as st

from acceuil import principale
from mainInvoice import mainInvoice
from mainCv import mainCV
import streamlit_authenticator as stauth  # pip install streamlit-authenticator

import database as db
# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Sales Dashboard", page_icon=":bar_chart:", layout="wide")



def acceuil():

    st.set_page_config(page_title="Cv Manipulation", page_icon="chart_with_upwards_trend", layout="wide")
    acceuil = ["Page Acceuil", "Invoice Parser", "Parser cv"]
    choice = st.sidebar.selectbox("Menu", acceuil)

    if choice == "Page Acceuil":
        print("acceuil")
        principale()

    elif choice == "Parser cv":
        mainCV()
    elif choice == "Invoice Parser":
        mainInvoice()
        print("invoice")

def first():
    # --- DEMO PURPOSE ONLY --- #
    placeholder = st.empty()
    placeholder.info("CREDENTIALS | username:pparker ; password:abc123")
    # ------------------------- #

    # --- USER AUTHENTICATION ---
    users = db.fetch_all_users()

    usernames = [user["key"] for user in users]
    names = [user["name"] for user in users]
    hashed_passwords = [user["password"] for user in users]

    authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
                                        "Extraction_from_files", "abcdef", cookie_expiry_days=30)

    name, authentication_status, username = authenticator.login("Login", "main")

    if authentication_status == False:
        st.error("Username/password is incorrect")

    if authentication_status == None:
        st.warning("Please enter your username and password")

    if authentication_status:
        placeholder.empty()
        acceuil()

        # ---- SIDEBAR ----
        authenticator.logout("Logout", "sidebar")
        st.sidebar.title(f"Welcome {name}")


first()
# acceuil()