# importing required libraries
# import datetime
# import yfinance as yf
# import dash
# from dash import dcc
# from dash import html
# from dash.dependencies import Input, Output
# import pandas_datareader.data as web


# app = dash.Dash()
# app.title = "Stock Visualisation"

# app.layout = html.Div(children=[
#     html.H1("Stock Visualisation Dashboard"),
#     html.H4("Please enter the stock name"),
#     dcc.Input(id='input', value='AAPL', type='text'),
#     html.Div(id='output-graph')
# ])

# # callback Decorator 
# @app.callback(
#     Output(component_id='output-graph', component_property='children'),
#     [Input(component_id='input', component_property='value')]
# )
# def update_graph(input_data):
#     start = datetime.datetime(2010, 1, 1)
#     end = datetime.datetime.now()

#     try:
#         df = web.DataReader(input_data, 'yahoo', start, end)

#         graph = dcc.Graph(id ="example", figure ={
#             'data':[{'x':df.index, 'y':df.Close, 'type':'line', 'name':input_data}],
#             'layout':{
#                 'title':input_data
#             }
#         })

#     except:
#         graph = html.Div("Error retrieving stock data.")

#     return graph

# if __name__ == '__main__':
#     app.run()



# def main() -> int:
#     """Main function to handle command line arguments."""


#     return 0

# if __name__ == '__main__':
    
#     sys.exit(main())  


# # Get / Generate data set for testing
# def getDataSet() -> tuple:
#     """This function returns a data set for testing."""
#     # df1 = pd.read_csv('out_ticks.csv')
#     # df2 = pd.read_csv('out_ohlcv.csv')
#     df1 = pd.read_csv('out_ticks.csv', index_col=0)
#     df2 = pd.read_csv('out_ohlcv.csv', index_col=0)
#     print(df1)
#     print(df2)
#     return df1, df2



# def getDataSet2() -> tuple:
#     # 1. Import the library
    

#     # 2. Tell the SEC who you are (required by SEC regulations)
#     # set_identity("your.name@example.com")  # Replace with your email

#     # 3. Find a company
#     # company = Company("MSFT")  # Microsoft
#     company = Company("FORD")  
#     print(company)

#     # 4. Get company filings
#     filings = company.get_filings()
#     print(filings)

#     # 5. Filter by form 
#     insider_filings = filings.filter(form="4")  # Insider transactions
#     print(insider_filings)

#     # 6. Get the latest filing
#     insider_filing = insider_filings[0]
#     print(insider_filing)

#     # 7. Convert to a data object
#     ownership = insider_filing.obj()
#     print(ownership)


# def getDataSet3() -> tuple:
#     company = Company("AAPL")
#     # financials = company.get_financials()
#     # print(financials)
    

#     filings = company.latest("10-K", 5)

#     # balance_sheet = financials.get_
#     # income_statement = financials.income
#     # cash_flow_statement = financials.cash_flow

#     financials = MultiFinancials(filings)
#     print(financials)
#     balance_sheet = financials.balance_sheet
#     income_statement = financials.income_statement
#     cash_flow_statement = financials.cashflow_statement
#     print(balance_sheet)
#     print(income_statement)
#     print(cash_flow_statement)


# def get_income_dataframe(ticker:str):
#     c = Company(ticker)
#     filings = c.get_filings(form="10-K").latest(5)
#     xbs = XBRLS.from_filings(filings)
#     income_statement = xbs.statements.income_statement()
#     income_df = income_statement.to_dataframe()
#     return income_df
    

# def plot_revenue(ticker:str):
#     c = Company(ticker)
#     income_df = get_income_dataframe(ticker)
#     # print(income_df.shape[0], income_df.shape[1])
#     # print(income_df.head())
#     # print(income_df)
#     # print(income_df.columns)
#     # print(income_df.to_period)
#     # print(income_df.index)
#     # print(income_df.columns[2:])  # Exclude the first two rows and the last row
    
#     # Extract financial metrics
#     net_income = income_df[income_df.concept == "us-gaap_NetIncomeLoss"].iloc[0]
#     # gross_profit = income_df[income_df.concept == "us-gaap_GrossProfit"].iloc[0]
#     # revenue = income_df[income_df.label == "Revenue"].iloc[0]

#     print("Net Income:", net_income)
#     # print("Gross Profit:", gross_profit)
#     # print("Revenue:", revenue)
    
#     # Convert periods to fiscal years for better readability
#     periods = [pd.to_datetime(period).strftime('FY%y') for period in income_df.columns[:][2:]]
    
#     # Reverse the order so most recent years are last (oldest to newest)
#     periods = periods[::-1]
#     # revenue_values = revenue.values[::-1]
#     # gross_profit_values = gross_profit.values[::-1]
#     net_income_values = net_income.values[:1:-1]

#     # print("Periods:", periods)
#     # print("Revenue:", revenue_values)
#     # print(revenue.shape)
#     # print("Gross Profit:", gross_profit_values)
#     # print("Net Income:", net_income_values)
    
#     # Create a DataFrame for plotting
#     plot_data = pd.DataFrame({
#         # 'Revenue': revenue_values,
#         # 'Gross Profit': gross_profit_values,
#         'Net Income': net_income_values
#     }, index=periods)
    
#     # Convert to billions for better readability
#     plot_data = plot_data / 1e9
    
#     # Create the figure
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     # Plot the data as lines with markers
#     plot_data.plot(kind='line', marker='o', ax=ax, linewidth=2.5)
    
#     # Format the y-axis to show billions with 1 decimal place
#     ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:.1f}B'))
    
#     # Add labels and title
#     ax.set_xlabel('Fiscal Year')
#     ax.set_ylabel('Billions USD')
#     ax.set_title(f'{c.name} ({ticker}) Financial Performance')
    
#     # Add a grid for better readability
#     ax.grid(True, linestyle='--', alpha=0.7)
    
#     # Add a source note
#     plt.figtext(0.5, 0.01, 'Source: SEC EDGAR via edgartools', ha='center', fontsize=9)
    
#     # Improve layout
#     plt.tight_layout(rect=[0, 0.03, 1, 0.97])

#     plt.show()
    
#     return fig


# def portfolio() -> None:
#     # Find a fund by ticker
#     # fund = find("VFIAX")  # Vanguard 500 Index Fund
#     fund = find("SWPPX")

#     # Get the fund's structure
#     classes = fund.get_classes()
#     print(f"Fund has {len(classes)} share classes")

#     # Get the latest portfolio holdings
#     portfolio = fund.get_portfolio()

#     # Show top 10 holdings by value
#     top_holdings = portfolio.sort_values('value', ascending=False).head(10)
#     print(top_holdings)
