import streamlit as st

from acceuil import principale
from mainInvoice import mainInvoice
from mainCv import mainCV


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




acceuil() 