import pandas as pd
import requests

class MarkupLoader:
    def __init__(self):
        # URL of the TableSense markup
        self.tablesense_url = "https://raw.githubusercontent.com/microsoft/TableSense/main/dataset/Table%20range%20annotations.txt"
        
    def get_markup(self, markup_name):
        if markup_name == "tablesense":
            response = requests.get(self.tablesense_url)
            data = response.content.decode('utf-8')
            
            # Load data into a pandas DataFrame
            columns = ['file_name', 'sheet_name', 'set_type', 'parent_path', 'table_range']
            df = pd.read_csv(pd.compat.StringIO(data), sep='\t', names=columns)
            return df
        
        else:
            print(f"Markup {markup_name} not found.")
            return None
