from tkinter import *
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.pyplot import figure
import pymysql as MySQLdb
from AI_trading_class import AlgorithmTrading
# Here, we are creating our class, Window, and inheriting from the Frame
# class. Frame is a class from the tkinter module. (see Lib/tkinter/__init__)

coin_dict =  {"BTC" : ["BTC", 'bitcoin' ],
              "ETH" : ["ETH" ,"ethereum"],
              "XRP": ["XRP", "ripple" ],
              "BCH": ["BCH", "Bitcoin-cash" ],
              "ADA": ["ADA", "cardano" ],
              "LTC": ["LTC", "liteCoin" ],
              "XEM": ["XEM", "nem" ],
              "NEO": ["NEO", "neo" ],
              "XLM": ["XLM", "stellar" ],
              "EOS": ["EOS", "eos" ],
              "MIOTA": ["MIOTA","iota" ],
              #"DASH": ['"DASH" OR "Dash" '],
              "XMR": ["XMR", "monero"],
              "TRX": ["TRX", "tron" ],
              "QASH": ["QASH", "QASH" ],
              "BTG": ["BTG", "bitcoin-gold"],
              "ICX": ["ICX", "icon" ],
              "QTUM": ["QTUM", "qtum" ],
              "ETC": ["ETC", "etheruem-classic"],
              "LSK": ["LSK", "lisk" ],
              #"NANO": ['"NANO"'],
              "VEN": ["VEN", "vechain" ],
              "OMG": ['"OMG" OR "OmiseGO" '],
              "PPT": ["PPT", "populous"],
              "XVG": ["XVG", "verge" ],
              "USDT": ["USDT", "tether" ],
              "XTZ": ["XTZ", "tezos"]}

models = ['svm', 'log', 'NN', 'Random Forest', 'Naive Bayes Gauss']

class Window(Frame):

    # Define settings upon initialization. Here you can specify
    def __init__(self, master=None):

        # parameters that you want to send through the Frame class.
        Frame.__init__(self, master)

        #reference to the master widget, which is the tk window
        self.master = master

        self.coins = 'Nothing'
        self.entered_coins = ''
        self.used_coins = []

        self.ratios = 'Nothing'
        self.entered_ratios = ''
        self.used_ratios = []

        self.money = 0
        self.entered_money = 0

        self.model = 'Nothing'
        self.entered_model = ''

        self.lag = 0
        self.entered_lag = ''

        self.date_time_1='Format:YYYY-MM-DD hh:mm:ss OPTIONAL!'
        self.date_time_2=''
        self.entered_date_1 = None
        self.entered_date_2 = None
        #with that, we want to then run init_window, which doesn't yet exist
        self.init_window()

    #Creation of init_window
    def init_window(self):
        self.master.title("GUI") # changing the title of our master widget
        #self.pack(fill=BOTH, expand=1) # allowing the widget to take the full space of the root window
        # creating a menu instance
        menu = Menu(self.master)
        self.master.config(menu=menu)
        # create the file object)
        file = Menu(menu)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        file.add_command(label="Exit", command=self.client_exit)

        #added "file" to our menu
        menu.add_cascade(label="File", menu=file)

        # create the file object)
        edit = Menu(menu)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        edit.add_command(label="Show Img", command=self.showImg)
        edit.add_command(label="Show Text", command=self.showText)

        #added "file" to our menu
        menu.add_cascade(label="Edit", menu=edit)

        self.coin_choices = Label(self.master, text="The choices for coins are:")
        self.coin_choices.grid(row=0, column=6, padx=6, pady=3, sticky=N+E+W+S)

        self.listBox_1 = Listbox(self.master, selectmode=BROWSE)
        self.listBox_1.grid(row=1, column=6, padx=6, rowspan=8)

        self.model_choices = Label(self.master, text="The coices for models are:")
        self.model_choices.grid(row=0, column=7, padx=6, pady=3, sticky=N+E+W+S)

        self.listBox_2 = Listbox(self.master, selectmode=BROWSE)
        self.listBox_2.grid(row=1, column=7, padx=6, rowspan=8)

        for item in models:
            self.listBox_2.insert("end", item)

        for item in coin_dict:
            self.listBox_1.insert("end", item)

        #Coin box
        self.coin_text = StringVar()
        self.coin_text.set(self.coins)
        self.list_coins = Label(self.master, textvariable=self.coin_text).grid(row=1, column=1, columnspan=2, padx=5, sticky=W)
        self.label_coins = Label(self.master, text='Coins:').grid(row=0)

        vcmd_coins = self.master.register(self.validate_coins)
        self.entry_coins = Entry(self.master, validate='key', validatecommand=(vcmd_coins, '%P'))
        self.submit_button_coins = Button(self.master, text="Submit", command=lambda: self.update("coins"))

        self.entry_coins.grid(row=0, column=1, columnspan=2, sticky=W+E)
        self.submit_button_coins.grid(row=0, column=5, rowspan=1, pady=5, padx=5)

        #Ratio box
        self.ratios_text = StringVar()
        self.ratios_text.set(self.ratios)
        self.list_ratios = Label(self.master, textvariable=self.ratios_text).grid(row=3, column=1, columnspan=2, padx=5, sticky=W)
        self.label_ratios = Label(self.master, text='Ratios:').grid(row=2)

        vcmd_ratios = self.master.register(self.validate_ratios)
        self.entry_ratios = Entry(self.master, validate='key', validatecommand=(vcmd_ratios, '%P'))
        self.submit_button_ratios = Button(self.master, text="Submit", command=lambda: self.update("ratios"))

        self.entry_ratios.grid(row=2, column=1, columnspan=2, sticky=W+E)
        self.submit_button_ratios.grid(row=2, column=5, rowspan=1, pady=5, padx=5)

        #money box
        self.money_text = IntVar()
        self.money_text.set(self.money)
        self.list_money = Label(self.master, textvariable=self.money_text).grid(row=5, column=1, columnspan=2, padx=5, sticky=W)
        self.label_money= Label(self.master, text='Money:').grid(row=4)

        vcmd_money = self.master.register(self.validate_money)
        self.entry_money = Entry(self.master, validate='key', validatecommand=(vcmd_money, '%P'))
        self.submit_button_money = Button(self.master, text='submit', command=lambda: self.update('money'))

        self.entry_money.grid(row=4, column=1, columnspan=2, sticky=W+E)
        self.submit_button_money.grid(row=4, column=5, rowspan=1, pady=5, padx=5)

        #Model box
        self.model_text = StringVar()
        self.model_text.set(self.model)
        self.list_model = Label(self.master, textvariable=self.model_text).grid(row=7, column=1, columnspan=2, padx=5, sticky=W)
        self.label_model = Label(self.master, text="Model:").grid(row=6)

        vcmd_model = self.master.register(self.validate_model)
        self.entry_model = Entry(self.master, validate='key', validatecommand=(vcmd_model, '%P'))
        self.submit_button_model = Button(self.master, text='submit', command=lambda: self.update('model'))

        self.entry_model.grid(row=6, column=1, columnspan=2, sticky=W+E)
        self.submit_button_model.grid(row=6, column=5, rowspan=1, pady=5, padx=5)

        #Lag box
        self.lag_text = IntVar()
        self.lag_text.set(self.lag)
        self.list_lag = Label(self.master, textvariable=self.lag_text).grid(row=9, column=1, columnspan=2, padx=5, sticky=W)
        self.label_lag = Label(self.master, text="Lag:").grid(row=8)

        vcmd_lag = self.master.register(self.validate_lag) # we have to wrap the command
        self.entry_lag = Entry(self.master, validate='key', validatecommand=(vcmd_lag, '%P'))
        self.submit_button_lag = Button(self.master, text='submit', command=lambda: self.update('lag'))

        self.entry_lag.grid(row=8, column=1, columnspan=2, sticky=W+E)
        self.submit_button_lag.grid(row=8, column=5, rowspan=1, pady=5, padx=5)

        #Datetime Box
        self.dates = StringVar()
        self.dates.set(self.date_time_1)
        self.list_dates = Label(self.master, textvariable=self.dates).grid(row=11, column=1, columnspan=2, padx=5, sticky=W)
        self.label_dates = Label(self.master, text="datetime:").grid(row=10)

        vcmd_date_1 = self.master.register(self.validate_date_1)
        vcmd_date_2 = self.master.register(self.validate_date_2)
        self.entry_dates_1 = Entry(self.master, validate='key', validatecommand=(vcmd_date_1, '%P'))
        self.entry_dates_2 = Entry(self.master, validate='key', validatecommand=(vcmd_date_2, '%P'))
        self.submit_button_dates = Button(self.master, text='submit', command=lambda: self.update('date'))

        self.entry_dates_1.grid(row=10, column=1, columnspan=1, padx=3, sticky=W+E)
        self.entry_dates_2.grid(row=10, column=2, columnspan=1, padx=3, sticky=W+E)
        self.submit_button_dates.grid(row=10, column=5, rowspan=1, pady=5, padx=5)

        #Create bot
        self.bot_label = Label(self.master, text="When all values are set you can click the button below!")
        self.button_bot_creation = Button(self.master, text='Create bot!', command=self.create_bot)

        self.bot_label.grid(row=12, column=1, columnspan=2, sticky=N+E+W+S)
        self.button_bot_creation.grid(row=13, column=1, columnspan=1, sticky=N+E+W+S)

    def showImg(self):
        load = Image.open("chat.png")
        render = ImageTk.PhotoImage(load)

        # labels can be text or images
        img = Label(self, image=render)
        img.image = render
        img.place(x=0, y=0)


    def showText(self):
        text = Label(self.master, text="Hey there good lookin!")
        text.grid(row=2, column=8)


    def validate_date_1(self, new_text):
        if not new_text:
            self.date_time_1 = None
            return True

        try:
            dates = str(new_text)
            self.entered_date_1 = dates
            return True
        except ValueError:
            return False

    def validate_date_2(self, new_text):
        if not new_text:
            self.date_time_2 = None
            return True

        try:
            dates = str(new_text)
            self.entered_date_2 = dates
            return True
        except ValueError:
            return False

    def validate_coins(self, new_text):
        if not new_text: # the field is being cleared
            self.coins = None
            return True

        try:
            coins = str(new_text)
            coins = coins.replace(" ", "")
            self.entered_coins = coins
            return True
        except ValueError:
            return False

    def validate_ratios(self, new_text):
        if not new_text: # the field is being cleared
            self.entered_ratios = None
            return True

        try:
            ratios = str(new_text)
            ratios = ratios.replace(" ", "")
            self.entered_ratios = ratios
            return True
        except ValueError:
            return False

    def validate_money(self, new_text):
        if not new_text: # the field is being cleared
            self.entered_money = None
            return True

        try:
            money = int(new_text)
            self.entered_money = money
            return True
        except ValueError:
            return False

    def validate_model(self, new_text):
        if not new_text: # the field is being cleared
            self.entered_model = None
            return True

        try:
            model = str(new_text)
            self.entered_model = model
            return True
        except ValueError:
            return False

    def validate_lag(self, new_text):
        if not new_text: # the field is being cleared
            self.entered_lag = None
            return True

        try:
            lag = int(new_text)
            self.entered_lag = lag
            return True
        except ValueError:
            return False

    def create_bot(self):
        try:
            if self.entered_date_1 == None:
                self.crypto_bot = AlgorithmTrading( self.used_coins,
                                                    self.used_ratios,
                                                    self.money,
                                                    self.model,
                                                    self.lag)


                self.newWindow = Toplevel(self.master)
                self.app = BotWindow(self.newWindow,
                                    self.used_coins,
                                    self.used_ratios,
                                    self.money,
                                    self.model,
                                    self.lag,
                                    self.crypto_bot,
                                    date=False)
            else:
                self.crypto_bot = AlgorithmTrading( self.used_coins,
                                                    self.used_ratios,
                                                    self.money,
                                                    self.model,
                                                    self.lag,
                                                    self.date_time_1,
                                                    self.date_time_2)


                self.newWindow = Toplevel(self.master)
                self.app = BotWindow(self.newWindow,
                                    self.used_coins,
                                    self.used_ratios,
                                    self.money,
                                    self.model,
                                    self.lag,
                                    self.crypto_bot,
                                    self.date_time_1,
                                    self.date_time_2,
                                    date=True)
        except ValueError as e:
            print("something went wrong")
            print(e)
            return False

    def update(self, method):
        if method == "coins":
            self.coins = self.entered_coins
            self.used_coins = self.entered_coins.split(",")
            self.coin_text.set(self.coins)
            self.entry_coins.delete(0, 'end')

        elif method == "ratios":
            self.ratios =self.entered_ratios
            self.used_ratios = self.entered_ratios.split(",")
            self.used_ratios = [float(i) for i in self.used_ratios]
            self.ratios_text.set(self.ratios)
            self.entry_ratios.delete(0, 'end')

        elif method == "money":
            self.money = self.entered_money
            self.money_text.set(self.money)
            self.entry_money.delete(0, 'end')

        elif method == "model":
            self.model = self.entered_model
            self.model_text.set(self.model)
            self.entry_model.delete(0, 'end')

        elif method == "lag":
            self.lag = self.entered_lag
            self.lag_text.set(self.lag)
            self.entry_lag.delete(0, 'end')

        elif method == "date":
            self.date_time_1 = self.entered_date_1
            self.date_time_2 = self.entered_date_2
            self.entry_dates_1.delete(0, 'end')
            self.entry_dates_2.delete(0, 'end')


    def client_exit(self):
        root.quit()
        root.destroy()


class BotWindow(Tk):
    def __init__(self, master, coins, ratios, money, model, lag, bot, date_1=None, date_2=None, date=False):
        self.master = master
        self.frame = Frame(self.master)
        #self.frame.grid_rowconfigure(0, weight=1)
        #self.frame.grid_columnconfigure(0, weight=1)

        self.coins = coins
        self.ratios = ratios
        self.money = money
        self.model = model
        self.lag = lag
        self.bot = bot
        self.date_1 = date_1
        self.date_2 = date_2

        self.init_bot_window()
        #self.frame.pack()
        self.frame.grid(row=0, column=0)

    def init_bot_window(self):
        self.master.title("Crypto Bot")
        self.master.geometry("800x400")
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_rowconfigure(0, weight=1)

        menu = Menu(self.master)
        self.master.config(menu=menu)
        # create the file object)
        file = Menu(menu)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        file.add_command(label="Exit", command=self.client_exit)

        #added "file" to our menu
        menu.add_cascade(label="File", menu=file)


        self.text = Text(self.frame, height=10, width=40)
        #self.text.pack(side='top', fill="both", expand=True)
        self.text.grid(row=0, column=0, rowspan=4, padx=5, sticky=N+E+W+S)

        self.text.delete("1.0", "end")
        self.text.insert("end", "Bot attributes are: \n")
        self.text.insert("end", "Coins is \t {} \n".format(self.coins) )
        self.text.insert("end", "Ratios are \t {} \n".format(self.ratios) )
        self.text.insert("end", "Money is \t {} \n".format(self.money) )
        self.text.insert("end", "Model used is \t {} \n".format(self.model) )
        self.text.insert("end", "The time lag is \t {} \n".format(self.lag) )
        if self.date_1 == None:
            self.text.insert("end", "starting date is \t {} \n".format("2018-06-25 08:00:00"))
            self.text.insert("end", "ending date is \t {} \n".format("2018-07-03 20:00:00"))
        else:
            self.text.insert("end", "starting date is \t {} \n".format(self.date_1))
            self.text.insert("end", "ending date is \t {} \n".format(self.date_2))
        self.text.insert("end", "\n" )
        self.text.bind("<Configure>", self.reset_tabstop)
        self.text.update_idletasks()


        self.button_frame = Frame(self.master)
        self.button_frame.grid(column=1, row=0)

        self.train_button = Button(self.button_frame, text='Train crypto bot', command=lambda: self.perform_class_function("train"))
        self.train_button.grid(row=0, column=0, pady=6, sticky=E+N)

        self.test_button = Button(self.button_frame, text='Perform testing', command=lambda: self.perform_class_function("testing"))
        self.test_button.grid(row=1, column=0, pady=6, sticky=E+N)

        self.confusion_button = Button(self.button_frame, text='Create Confusion matrix', command=lambda: self.perform_class_function("confusion"))
        self.confusion_button.grid(row=2, column=0, pady=6, sticky=E+N)

        self.CV_button = Button(self.button_frame, text='Perform Cross Validation', command=lambda: self.perform_class_function("CV"))
        self.CV_button.grid(row=3, column=0, pady=6, sticky=E+N)

    def reset_tabstop(self, event):
        event.widget.configure(tabs=(event.width-8, "right"))

    def perform_class_function(self, method):
        if method =="train":
            self.bot.trading_train()
            self.text.insert("end", "Model trained:YES \n")
            self.text.update_idletasks()
        elif method =="CV":
            fig = self.bot.perform_CV()
            fig.set_size_inches(w=4, h=2.67, forward=True)
            canvas = FigureCanvasTkAgg(fig, master=self.master)
            canvas.show()
            canvas.get_tk_widget().grid(row=1, column=4, columnspan=2, padx=5, pady=7)
        elif method =="testing":
            fig = self.bot.trading_test()
            fig.set_size_inches(w=4, h=2.67, forward=True)
            canvas = FigureCanvasTkAgg(fig, master=self.master)
            canvas.show()
            canvas.get_tk_widget().grid(row=1, column=0, columnspan=2, padx=5, pady=7)
        elif method =="confusion":
            fig = self.bot.plot_confusion_matrix()
            fig.set_size_inches(w=4, h=2.67, forward=True)
            canvas = FigureCanvasTkAgg(fig, master=self.master)
            canvas.show()
            canvas.get_tk_widget().grid(row=1, column=2, columnspan=2, padx=5, pady=7)

    def client_exit(self):
        root.quit()
        root.destroy()
# root window created. Here, that would be the only window, but
# you can later have windows within windows.
root = Tk()

root.geometry("800x400")

#creation of an instance
app = Window(root)

#mainloop
root.mainloop()
