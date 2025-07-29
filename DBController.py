import psycopg2
import streamlit as st


def get_connection():
    return psycopg2.connect(
        host=st.secrets["db_host"],
        port="5432",
        dbname=st.secrets["db_name"],
        user=st.secrets["db_user"],
        password=st.secrets["db_password"]
    )
