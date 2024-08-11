import wx

"""
This module provides a GUI for some parts of Featuretools. 
"""

class MyPanel(wx.Panel):
    def __init__(self, parent=None):
        super(MyPanel, self).__init__(parent)

        # Label 1
        self.label = wx.StaticText(self, wx.ID_ANY, " Featuretools - Choose a feature ")
        self.label.SetPosition((50, 20))  # Set position manually
        
        # Create first button
        self.button = wx.Button(self, wx.ID_ANY, "Test Button")
        self.button.SetPosition((50, 60))  # Set position manually
        
        # Bind event handler for the first button
        self.Bind(wx.EVT_BUTTON, self.onButtonClicked, self.button)

    def onButtonClicked(self, event):
        
        """Action when button is clicked """

        print("Button clicked")

class MyApp(wx.App):
    def OnInit(self):
        frame = wx.Frame(None, wx.ID_ANY, "Panel Example")
        panel = MyPanel(frame)
        frame.Show()
        return True

if __name__ == "__main__":
    app = MyApp()
    app.MainLoop()
