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
            data = response.content.decode('utf-8')
            
            # Load data into a pandas DataFrame
            df = pd.read_csv(io.StringIO(data), sep='\t', header=None)
            
            # Process the DataFrame to combine table_range values into a list
            processed_data = []
            for _, row in df.iterrows():
                # The first 4 columns are file_name, sheet_name, set_type, and parent_path
                file_name, sheet_name, set_type, parent_path = row[:4]
                # All remaining columns are table_range values
                table_ranges = row[4:].dropna().tolist()
                processed_data.append([file_name, sheet_name, set_type, parent_path, table_ranges])
            
            # Convert the processed data into a DataFrame
            columns = ['file_name', 'sheet_name', 'set_type', 'parent_path', 'table_range']
            df = pd.DataFrame(processed_data, columns=columns)
            return df
            
        else:
            print(f"Markup {markup_name} not found.")
            return None
