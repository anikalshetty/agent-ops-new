import pandas as pd
from DBController import get_connection
from datetime import timedelta

class LLMQueries:

    def get_dropdown_options(self):
        query = "SELECT DISTINCT name as category FROM projects ORDER BY name"
        with get_connection() as conn:
            df = pd.read_sql(query, conn)
        return df['category'].tolist()

    def get_tile1_value(self, selected, start_date, end_date):
        query = """
        SELECT COUNT(*) AS value
        FROM spans
        WHERE span_kind = 'LLM'
          AND start_time BETWEEN %s AND %s
          AND trace_rowid IN (
              SELECT id FROM traces WHERE project_rowid IN (
                  SELECT id FROM projects WHERE name = %s
              )
           )   
        """
        with get_connection() as conn:
            return pd.read_sql(query, conn, params=(start_date, end_date, selected))['value'][0]

    def get_tile2_value(self, selected, start_date, end_date):
        query = """
        SELECT AVG(EXTRACT(EPOCH FROM end_time - start_time)) AS value 
        FROM spans 
        WHERE start_time BETWEEN %s AND %s
          AND trace_rowid IN (
              SELECT id FROM traces 
              WHERE project_rowid IN (
                  SELECT id FROM projects WHERE name = %s
              )
          )
        """
        with get_connection() as conn:
            return pd.read_sql(query, conn, params=(start_date, end_date, selected))['value'][0]

    def get_tile3_value(self, selected, start_date, end_date):
        query = """
        SELECT SUM(llm_token_count_prompt) AS value 
        FROM spans 
        WHERE start_time BETWEEN %s AND %s
          AND trace_rowid IN (
              SELECT id FROM traces 
              WHERE project_rowid IN (
                  SELECT id FROM projects WHERE name = %s
              )
          )
        """
        with get_connection() as conn:
            return pd.read_sql(query, conn, params=(start_date, end_date, selected))['value'][0]

    def get_tile4_value(self, selected, start_date, end_date):
        query = """
        SELECT SUM(llm_token_count_completion) AS value 
        FROM spans 
        WHERE start_time BETWEEN %s AND %s
          AND trace_rowid IN (
              SELECT id FROM traces 
              WHERE project_rowid IN (
                  SELECT id FROM projects WHERE name = %s
              )
          )
        """
        with get_connection() as conn:
            return pd.read_sql(query, conn, params=(start_date, end_date, selected))['value'][0]

    def get_metric1_data(self, selected, start_date, end_date):
        query = """
        SELECT DATE_TRUNC('hour', start_time) as hour,
               COUNT(DISTINCT trace_rowid) AS count
        FROM spans
        WHERE span_kind = 'LLM'
          AND start_time BETWEEN %s AND %s
          AND trace_rowid IN (
              SELECT id FROM traces
              WHERE project_rowid IN (
                  SELECT id FROM projects WHERE name = %s
              )
          )
        GROUP BY hour
        ORDER BY hour;
        """
        with get_connection() as conn:
            return pd.read_sql(query, conn, params=(start_date, end_date, selected))

    def get_metric2_data(self, selected, start_date, end_date):
        query = """
        SELECT DATE(start_time) AS date,
       status_code AS status,
       COUNT(*) AS trace_count
        FROM spans
        WHERE start_time BETWEEN %s AND %s
        AND trace_rowid IN (
              SELECT id FROM traces
              WHERE project_rowid IN (
                  SELECT id FROM projects WHERE name = %s
              )
         )     
        GROUP BY date, status
        ORDER BY date, status;
        """
        with get_connection() as conn:
            return pd.read_sql(query, conn, params=(start_date, end_date, selected))

    def get_trace_by_name(self, selected, start_date, end_date):
        query = """
        SELECT name,
       COUNT(*) AS trace_count
        FROM spans
        WHERE start_time BETWEEN %s AND %s
        AND trace_rowid IN (
              SELECT id FROM traces
              WHERE project_rowid IN (
                  SELECT id FROM projects WHERE name = %s
              )
         )     
        GROUP BY name
        ORDER BY name;
        """
        with get_connection() as conn:
            return pd.read_sql(query, conn, params=(start_date, end_date, selected))

    def get_root_spans(self, project_name):
        query = """
        SELECT s.trace_rowid, s.span_id, s.name, s.start_time
        FROM spans s
        JOIN traces t ON s.trace_rowid = t.id
        WHERE s.parent_id IS NULL
          AND t.project_rowid = (
            SELECT id FROM projects WHERE name = %s
          )
        ORDER BY s.start_time DESC
        """
        with get_connection() as conn:
            return pd.read_sql(query, conn, params=(project_name,))

    def get_span_tree_for_trace(self, trace_id):
        query = """
        SELECT span_id, parent_id, name,
               EXTRACT(EPOCH FROM (end_time - start_time)) AS latency
        FROM spans
        WHERE trace_rowid = %s
        ORDER BY start_time
        """
        with get_connection() as conn:
            return pd.read_sql(query, conn, params=(trace_id,))

    def get_expensive_spans(self, project_name, start_date, end_date):
        query = """
        SELECT s.span_id,
               s.name,
               EXTRACT(EPOCH FROM (s.end_time - s.start_time)) AS latency,
               s.llm_token_count_prompt AS prompt_tokens,
               s.llm_token_count_completion AS completion_tokens,
               (COALESCE(s.llm_token_count_prompt, 0) + COALESCE(s.llm_token_count_completion, 0)) AS total_tokens
        FROM spans s
        JOIN traces t ON s.trace_rowid = t.id
        WHERE t.project_rowid = (
            SELECT id FROM projects WHERE name = %s
        )
          AND s.start_time BETWEEN %s AND %s
        ORDER BY total_tokens DESC, latency DESC
        """
        with get_connection() as conn:
            return pd.read_sql(query, conn, params=(project_name, start_date, end_date))

    def get_date_range_for_project(self, project_name):
        query = """
        SELECT MIN(start_time) AS min_date,
               MAX(start_time) AS max_date
        FROM spans
        WHERE trace_rowid IN (
            SELECT id FROM traces
            WHERE project_rowid IN (
                SELECT id FROM projects WHERE name = %s
            )
        )
        """
        with get_connection() as conn:
            df = pd.read_sql(query, conn, params=(project_name,))
            return df['min_date'][0].date(), df['max_date'][0].date()+timedelta(days=1)
        
        
    # def getAttributes(self,selected,start_date, end_date):
    #  query = """
    #     SELECT p.name,s.name as span_name, s.start_time,s.attributes, s. span_kind
    #     FROM projects p
    #     JOIN traces t ON p.id = t.project_rowid
    #     JOIN spans s ON t.id = s.trace_rowid
    #     WHERE s.attributes::text ILIKE %s
    #     AND p.name = %s
    #     AND s.start_time BETWEEN %s AND %s
    #     """
    #  with get_connection() as conn:
    #     return pd.read_sql(query, conn, params=("%%gpt-%%", selected,start_date, end_date))
 
    def getAttributes(self,selected,start_date, end_date):
     query = """
        SELECT p.name,s.trace_rowid,s.name as span_name, s.start_time,s.end_time,s.attributes, s. span_kind, 
        s.span_id,s.parent_id,t.start_time as timestamp,
        s.cumulative_error_count,s.cumulative_llm_token_count_prompt,s.cumulative_llm_token_count_completion,s.status_code, s.status_message
        FROM projects p
        JOIN traces t ON p.id = t.project_rowid
        JOIN spans s ON t.id = s.trace_rowid
        WHERE p.name = %s
        AND s.start_time BETWEEN %s AND %s
        """
     with get_connection() as conn:
        return pd.read_sql(query, conn, params=( selected,start_date, end_date))

