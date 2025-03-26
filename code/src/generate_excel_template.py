import pandas as pd
import numpy as np

def create_sample_template():
    # Sample data
    data = {
        'Customer ID': ['CUST001', 'CUST002', 'CUST003'],
        'Age': [25, 34, 45],
        'Gender': ['Male', 'Female', 'Male'],
        'Purchase History': [
            'Laptop, Smartphone, Headphones',
            'Tablet, Smart Watch, Fitness Band',
            'Desktop PC, Gaming Console, VR Headset'
        ],
        'Interests': [
            'Technology, Gaming, Music',
            'Fitness, Health, Technology',
            'Gaming, Virtual Reality, Programming'
        ],
        'Engagement score': [150, 180, 120],
        'Sentiment Score': [0.8, 0.6, 0.9],
        'Social media activity level': ['High', 'Med', 'Low']
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add column descriptions
    template_info = pd.DataFrame({
        'Column Name': df.columns,
        'Description': [
            'Unique identifier for each customer',
            'Customer age in years',
            'Customer gender',
            'Comma-separated list of past purchases',
            'Comma-separated list of customer interests',
            'Engagement level (0-200)',
            'Sentiment analysis score (-1 to 1)',
            'Social media activity (Low/Med/High)'
        ],
        'Example': [
            'CUST001',
            '25',
            'Male/Female',
            'Laptop, Smartphone, Headphones',
            'Technology, Gaming, Music',
            '150',
            '0.8',
            'High'
        ]
    })
    
    # Create Excel writer
    with pd.ExcelWriter('user_data_template.xlsx', engine='openpyxl') as writer:
        # Write sample data
        df.to_excel(writer, sheet_name='Sample Data', index=False)
        
        # Write template info
        template_info.to_excel(writer, sheet_name='Instructions', index=False)
        
        # Auto-adjust column widths
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column = [cell for cell in column]
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = (max_length + 2)
                worksheet.column_dimensions[column[0].column_letter].width = adjusted_width

if __name__ == '__main__':
    create_sample_template()
    print("Template created successfully as 'user_data_template.xlsx'")
