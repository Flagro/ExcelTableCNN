import pandas as pd
import requests
import io

class MarkupLoader:
    def __init__(self):
        # URL of the TableSense markup
        self.tablesense_url = "https://raw.githubusercontent.com/microsoft/TableSense/main/dataset/Table%20range%20annotations.txt"
        
    def get_markup(self, markup_name):
        if markup_name == "tablesense":
            response = requests.get(self.tablesense_url)
            lines = response.content.decode('utf-8').splitlines()
            
            # Process each line to split on tab and organize data
            processed_data = []
            for line in lines:
                fields = line.split('\t')
                
                # The first 4 fields are file_name, sheet_name, set_type, and parent_path
                file_name, sheet_name, set_type, parent_path = fields[:4]
                parent_path = parent_path.replace("\\", "/")
                # All remaining fields are table_range values
                table_ranges = fields[4:]
                processed_data.append([file_name, sheet_name, set_type, parent_path, table_ranges])
            
            # Convert the processed data into a DataFrame
            columns = ['file_name', 'sheet_name', 'set_type', 'parent_path', 'table_range']
            df = pd.DataFrame(processed_data, columns=columns)
            return df
            
        else:
            print(f"Markup {markup_name} not found.")
            return None
