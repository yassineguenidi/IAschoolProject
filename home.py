import streamlit as st
st.set_page_config(page_title="Cv Manipulation", page_icon="chart_with_upwards_trend", layout="wide")

from acceuil import principale
from mainInvoice import mainInvoice
from mainCv import mainCV


def acceuil():
    menu = ["Page Acceuil", "Invoice Parser", "Parser cv"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Page Acceuil":
        principale()
    elif choice == "Parser cv":
        mainCV()
    elif choice == "Invoice Parser":
        mainInvoice()

acceuil()
