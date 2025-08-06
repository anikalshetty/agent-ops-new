# Agent-Ops Streamlit App

This is a simple Streamlit application instrumented with AgentOps for observability of agent workflows.

## 🔧 Running Locally

### 1. **Clone the repository**

```bash
git clone https://github.com/0xSushmitha/agent-ops-new.git
cd agent-ops-new
```

### 2. **Create and activate a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate      # On Windows use: venv\\Scripts\\activate
```

### 3. **Install dependencies**

```bash
pip install -r requirements.txt
```

### 4. **Set DB secrets**

#### In DBController.py if local
  ```
  def get_connection():
      return psycopg2.connect(
          host="",
          port="5432",
          dbname="",
          user="",
          password=""
      )
  ```
#### In streamlit cloud if hosted in cloud
  ```
  db_host=""
  db_name=""
  db_user=""
  db_password=""
  ```

### 5. **Run the app**

   ```
   streamlit run app.py
   ```

## Online hosted app

[https://agent-ops-new.streamlit.app/](https://agent-ops-new.streamlit.app/)


