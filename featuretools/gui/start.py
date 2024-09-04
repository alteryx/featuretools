"""
This module provides a GUI to merge 2 CSV-files containing data-sets into one 
using deep feature sythesis.
"""

import wx
import pandas as pd
import featuretools as ft
import warnings

# Disable all warnings
warnings.filterwarnings("ignore")


VERT_SPACE = 40
LEFT_MARGIN = 10


class Frame(wx.Frame):

    """
    This class contains all the GUI-functions
    """

    def __init__(self, parent, title):
        super().__init__(parent, title=title, size=(600, 800))

        # Create panel
        self.panel = wx.Panel(self)

        # Create variables
        self.filepath_1 = ""
        self.filepath_2 = ""

        # Directory to save csv-file
        self.directory = ""

        # Text area for primary keys
        self.text_area_1 = wx.TextCtrl(self.panel)
        self.text_area_2 = wx.TextCtrl(self.panel)

        description_1 = "Select 2 CSV-files and merge them into 1 using deep feature synthesis\n"
        description_2 = "The primary key of file 1 must be also present in file 2"

        # Description labels
        self.general_description = wx.StaticText(
            self.panel, label=description_1+description_2)
        self.user_information = wx.StaticText(self.panel, label="")
        self.path_1_label = wx.StaticText(self.panel, label="")
        self.key_1_label = wx.StaticText(
            self.panel, label="Primary key of file 1:")
        self.path_2_label = wx.StaticText(self.panel, label="")
        self.key_2_label = wx.StaticText(
            self.panel, label="Primary key of file 2:")

        # Create static text control
        self.file_name_label = wx.StaticText(self.panel, label="")

        # Create and bind buttons
        self.pick_1_button = wx.Button(self.panel, label='Pick file 1:')
        self.Bind(wx.EVT_BUTTON, self.on_pick_file_1, self.pick_1_button)

        self.pick_2_button = wx.Button(self.panel, label='Pick file 2:')
        self.Bind(wx.EVT_BUTTON, self.on_pick_file_2, self.pick_2_button)

        self.continue_button = wx.Button(self.panel, label="Continue")
        self.Bind(wx.EVT_BUTTON, self.continue_pressed, self.continue_button)

        # Layout control
        self.general_description.SetPosition((LEFT_MARGIN, VERT_SPACE*1))

        self.pick_1_button.SetPosition((LEFT_MARGIN, VERT_SPACE*4))
        self.path_1_label.SetPosition((LEFT_MARGIN, VERT_SPACE*5))
        self.key_1_label.SetPosition((LEFT_MARGIN, VERT_SPACE*6))
        self.text_area_1.SetPosition((LEFT_MARGIN, VERT_SPACE*7))

        self.pick_2_button.SetPosition((LEFT_MARGIN, VERT_SPACE*9))
        self.path_2_label.SetPosition((LEFT_MARGIN, VERT_SPACE*10))
        self.key_2_label.SetPosition((LEFT_MARGIN, VERT_SPACE*11))
        self.text_area_2.SetPosition((LEFT_MARGIN, VERT_SPACE*12))

        self.continue_button.SetPosition((LEFT_MARGIN, VERT_SPACE*14))
        self.user_information.SetPosition((LEFT_MARGIN, VERT_SPACE*16))

    def continue_pressed(self, event):
        """
        Action when continue-button is pressed
        First asks for a directory to save the result,
        then the data-processing starts
        """

        if not self.check_fields():
            self.user_information.SetLabel("Error: Missing values")
            return False

        with wx.DirDialog(self, "Choose a directory to save the new file", style=wx.DD_DEFAULT_STYLE) as dialog:
            if dialog.ShowModal() == wx.ID_OK:
                self.directory = dialog.GetPath()

                # Error handling
                try:
                    self.process_data()
                    self.user_information.SetLabel(
                        "File result.csv has been generated")
                except:
                    self.user_information.SetLabel(
                        " Error: Wrong information \n Check keys and dataset")

    def process_data(self):
        """The acutal data processing and saving"""

        # Load the data into dataframes
        df1 = pd.read_csv(self.filepath_1)
        df2 = pd.read_csv(self.filepath_2)

        # Load primary keys:
        pk1 = self.text_area_1.GetValue()
        pk2 = self.text_area_2.GetValue()

        # Create a new EntitySet
        entities = {
            "data1": (df1, pk1),
            "data2": (df2, pk2),
        }

        # Add a relationship between datasets
        relationships = [("data1", pk1, "data2", pk1)]

        # Run deep feature synthesis to automatically generate features
        feature_matrix, features_defs = ft.dfs(
            dataframes=entities,
            relationships=relationships,
            target_dataframe_name="data1",
        )

        print(feature_matrix)

        dir_and_filename = self.directory+'result.csv'

        feature_matrix.to_csv(dir_and_filename)

    def on_pick_file_1(self, event):
        self.pick_file(1)

    def on_pick_file_2(self, event):
        self.pick_file(2)

    def pick_file(self, filenumber):
        """File dialog"""

        with wx.FileDialog(self, "Open a file", wildcard="CSV files (*.csv)|*.csv",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_OK:
                # Get the selected file path
                filepath = fileDialog.GetPath()

            if filenumber == 1:
                self.filepath_1 = filepath
                self.path_1_label.SetLabel(filepath)
            else:
                self.filepath_2 = filepath
                self.path_2_label.SetLabel(filepath)

    def check_fields(self):
        """Checks if necessary fields are empty"""

        if self.text_area_1 == "" or self.text_area_2 == "":
            return False
        if self.filepath_1 == "" or self.filepath_2 == "":
            return False

        return True


if __name__ == '__main__':
    app = wx.App()
    frame = Frame(None, title="Feature Tools")
    frame.Show()
    app.MainLoop()
