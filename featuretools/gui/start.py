import wx

class Frame(wx.Frame):
    def __init__(self, parent, title):
        super().__init__(parent, title=title, size=(400, 300)) 
        
        #Create panels
        self.panel1 = wx.Panel(self)

        # List of chosen files
        self.chosen_files = [] 

        # Primary key
        self.primary_key = ""

        # Text area for primary key
        self.text_area = wx.TextCtrl(self.panel1)
        
        # Create file picker
        self.file_picker = wx.FilePickerCtrl(self.panel1, style=wx.FLP_DEFAULT_STYLE)

        # Create description labels
        self.description_label_1 = wx.StaticText(self.panel1, label="Choose multiple files:")

        self.description_label_2 = wx.StaticText(self.panel1, label="Choose a primary key:")
        
        # Create static text control
        self.file_name_label = wx.StaticText(self.panel1, label="")

        # Create and bind button
        self.button1 = wx.Button(self.panel1, label="Continue")
        self.Bind(wx.EVT_BUTTON, self.button_pressed, self.button1)
        
        # Layout controls manually
        self.description_label_1.SetPosition((10,10))
        self.file_picker.SetPosition((10, 40))
        self.button1.SetPosition((150,40))
        self.description_label_2.SetPosition((10,80))
        self.text_area.SetPosition((10,120))
        self.file_name_label.SetPosition((10, 160))
        
        # Bind file picker change event
        self.file_picker.Bind(wx.EVT_FILEPICKER_CHANGED, self.on_file_changed)
        

    def on_file_changed(self, event):
        file_path = self.file_picker.GetPath()
        self.chosen_files.append(file_path)
        self.file_name_label.SetLabel('\n'.join(self.chosen_files))
    
    def button_pressed(self,event):
        self.primary_key = self.text_area.GetValue()
    

if __name__ == '__main__':
    app = wx.App()
    frame = Frame(None, title="Feature Tools")
    frame.Show()
    app.MainLoop()
