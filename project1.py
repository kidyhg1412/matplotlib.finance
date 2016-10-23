## Project 1 获取数据并作折线图
import matplotlib.pyplot as plt
from matplotlib.finance import quotes_historical_yahoo_ochl  
from matplotlib.dates import DateFormatter  
import datetime  

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

ticker = '601985.ss'
date1 = datetime.date( 2015, 10, 1 )  
date2 = datetime.date( 2016, 10, 1 )  
  
daysFmt  = DateFormatter('%m-%d-%Y')  
  
quotes = quotes_historical_yahoo_ochl(ticker, date1, date2)  
if len(quotes) == 0:  
    raise SystemExit
print(quotes[1])

dates = [q[0] for q in quotes]  
opens = [q[1] for q in quotes]
closes = [q[2] for q in quotes]
  
fig = plt.figure()  
ax = fig.add_subplot(111)  
ax.plot_date(dates, opens, '-')  
  
# format the ticks  
ax.xaxis.set_major_formatter(daysFmt)  
ax.autoscale_view()  
  
# format the coords message box  
def price(x): 
    return '$%1.2f'%x 
    
ax.fmt_xdata = DateFormatter('%Y-%m-%d')  
ax.fmt_ydata = price  
ax.grid(True)

fig.autofmt_xdate()  
plt.title('中国核电 601985')
plt.show()

# Output: (735873.0, 8.9499999999999993, 8.9499999999999993, 8.9499999999999993, 8.9499999999999993, 0.0)
