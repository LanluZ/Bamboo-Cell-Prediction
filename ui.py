from tkinter import *
from tkinter.ttk import *


class WinGUI(Tk):
    def __init__(self):
        super().__init__()
        self.__win()
        self.tk_select_box_lupl6jkf = self.__tk_select_box_lupl6jkf(self)
        self.tk_button_luplah5d = self.__tk_button_luplah5d(self)
        self.tk_input_luplb537 = self.__tk_input_luplb537(self)
        self.tk_label_luplczy7 = self.__tk_label_luplczy7(self)
        self.tk_label_luplm8kp = self.__tk_label_luplm8kp(self)
        self.tk_text_luplnow9 = self.__tk_text_luplnow9(self)

    def __win(self):
        self.title("Model Perdictor")
        # 设置窗口大小、居中
        width = 350
        height = 160
        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()
        geometry = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.geometry(geometry)

        self.resizable(width=False, height=False)

    def scrollbar_autohide(self, vbar, hbar, widget):
        """自动隐藏滚动条"""

        def show():
            if vbar: vbar.lift(widget)
            if hbar: hbar.lift(widget)

        def hide():
            if vbar: vbar.lower(widget)
            if hbar: hbar.lower(widget)

        hide()
        widget.bind("<Enter>", lambda e: show())
        if vbar: vbar.bind("<Enter>", lambda e: show())
        if vbar: vbar.bind("<Leave>", lambda e: hide())
        if hbar: hbar.bind("<Enter>", lambda e: show())
        if hbar: hbar.bind("<Leave>", lambda e: hide())
        widget.bind("<Leave>", lambda e: hide())

    def v_scrollbar(self, vbar, widget, x, y, w, h, pw, ph):
        widget.configure(yscrollcommand=vbar.set)
        vbar.config(command=widget.yview)
        vbar.place(relx=(w + x) / pw, rely=y / ph, relheight=h / ph, anchor='ne')

    def h_scrollbar(self, hbar, widget, x, y, w, h, pw, ph):
        widget.configure(xscrollcommand=hbar.set)
        hbar.config(command=widget.xview)
        hbar.place(relx=x / pw, rely=(y + h) / ph, relwidth=w / pw, anchor='sw')

    def create_bar(self, master, widget, is_vbar, is_hbar, x, y, w, h, pw, ph):
        vbar, hbar = None, None
        if is_vbar:
            vbar = Scrollbar(master)
            self.v_scrollbar(vbar, widget, x, y, w, h, pw, ph)
        if is_hbar:
            hbar = Scrollbar(master, orient="horizontal")
            self.h_scrollbar(hbar, widget, x, y, w, h, pw, ph)
        self.scrollbar_autohide(vbar, hbar, widget)

    def __tk_select_box_lupl6jkf(self, parent):
        cb = Combobox(parent, state="readonly", )
        cb['values'] = ("time-feature", "feature-time")
        cb.place(x=20, y=10, width=150, height=30)
        return cb

    def __tk_button_luplah5d(self, parent):
        btn = Button(parent, text="预测", takefocus=False, )
        btn.place(x=20, y=110, width=150, height=35)
        return btn

    def __tk_input_luplb537(self, parent):
        ipt = Entry(parent, )
        ipt.place(x=20, y=60, width=150, height=30)
        return ipt

    def __tk_label_luplczy7(self, parent):
        label = Label(parent, text="模型", anchor="center", )
        label.place(x=200, y=10, width=100, height=30)
        return label

    def __tk_label_luplm8kp(self, parent):
        label = Label(parent, text="起始参数", anchor="center", )
        label.place(x=200, y=60, width=100, height=30)
        return label

    def __tk_text_luplnow9(self, parent):
        text = Text(parent)
        text.place(x=200, y=110, width=101, height=31)
        return text


class Win(WinGUI):
    def __init__(self, controller):
        self.ctl = controller
        super().__init__()
        self.__event_bind()
        self.__style_config()
        self.ctl.init(self)

    def __event_bind(self):
        self.tk_button_luplah5d.bind('<Button>', self.ctl.button_click_mouse)
        print(1)
        pass

    def __style_config(self):
        pass


if __name__ == "__main__":
    win = WinGUI()
    win.mainloop()
