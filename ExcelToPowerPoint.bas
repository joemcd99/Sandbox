Attribute VB_Name = "ExcelToPowerPoint"
Option Explicit

' ══════════════════════════════════════════════════════════════════════════
'  Excel → PowerPoint
'  Copies each configured sheet as a pixel-perfect picture into a new
'  PowerPoint presentation, preserving all formatting, charts, and layout.
'
'  HOW TO USE:
'    1. In Excel: Alt+F11 → File → Import File → select this .bas file
'    2. Edit SheetConfig() below with your sheet names and titles
'    3. Run ExportToPowerPoint  (Alt+F8 → ExportToPowerPoint → Run)
'
'  Requirements: Microsoft PowerPoint must be installed.
'  Tested on: Excel/PowerPoint 2016, 2019, 2021, Microsoft 365 (Windows)
' ══════════════════════════════════════════════════════════════════════════


' ── CONFIGURATION ──────────────────────────────────────────────────────────

' Each entry: Array("SheetName", "Slide Title")
' Set title to "" to skip the title bar on that slide.
Private Function SheetConfig() As Variant
    SheetConfig = Array( _
        Array("Sales Summary",  "Sales Summary"), _
        Array("Revenue Chart",  "Revenue Analysis"), _
        Array("KPIs",           "Key Performance Indicators") _
    )
End Function

' Output path for the .pptx file.
' Leave blank ("") to auto-save next to the workbook as <WorkbookName>_slides.pptx
Private Const OUTPUT_PATH As String = ""

' Show a navy title bar at the top of each slide?
Private Const SHOW_TITLE As Boolean = True

' Title bar height in points (1 inch = 72 pt)
Private Const TITLE_BAR_HEIGHT As Single = 34

' Padding around the content image on each side (points)
Private Const MARGIN As Single = 16

' ── END CONFIGURATION ──────────────────────────────────────────────────────


Public Sub ExportToPowerPoint()

    Dim cfg       As Variant
    Dim PPTApp    As Object
    Dim PPTPres   As Object
    Dim PPTSlide  As Object
    Dim sh        As Object          ' Worksheet or Chart sheet
    Dim entry     As Variant
    Dim slideIdx  As Long
    Dim outPath   As String

    cfg = SheetConfig()

    ' ── Launch / connect to PowerPoint ───────────────────────────────────
    On Error Resume Next
    Set PPTApp = GetObject(, "PowerPoint.Application")
    On Error GoTo 0

    If PPTApp Is Nothing Then
        Set PPTApp = CreateObject("PowerPoint.Application")
        PPTApp.Visible = True
    End If

    Set PPTPres = PPTApp.Presentations.Add(WithWindow:=True)

    ' Widescreen 16:9  (960 × 540 pt  =  13.33" × 7.5")
    With PPTPres.PageSetup
        .SlideWidth  = 960
        .SlideHeight = 540
    End With

    slideIdx = 0

    ' ── Loop over configured sheets ───────────────────────────────────────
    For Each entry In cfg

        Dim sheetName  As String
        Dim slideTitle As String
        sheetName  = CStr(entry(0))
        slideTitle = CStr(entry(1))

        ' Try to get the sheet (works for both worksheets and chart sheets)
        On Error Resume Next
        Set sh = ThisWorkbook.Sheets(sheetName)
        On Error GoTo 0

        If sh Is Nothing Then
            MsgBox "Sheet """ & sheetName & """ not found — skipping.", _
                   vbExclamation, "ExportToPowerPoint"
            GoTo ContinueLoop
        End If

        slideIdx = slideIdx + 1

        ' Add a blank slide  (ppLayoutBlank = 12)
        Set PPTSlide = PPTPres.Slides.Add(slideIdx, 12)
        Call ClearPlaceholders(PPTSlide)

        ' ── Optional title bar ───────────────────────────────────────────
        Dim topOffset As Single
        topOffset = MARGIN

        If SHOW_TITLE And slideTitle <> "" Then
            Call AddTitleBar(PPTSlide, slideTitle, _
                             PPTPres.PageSetup.SlideWidth, TITLE_BAR_HEIGHT)
            topOffset = TITLE_BAR_HEIGHT + MARGIN
        End If

        ' ── Copy the sheet content as a high-res picture ─────────────────
        Application.ScreenUpdating = False

        If sh.Type = xlChart Then
            ' ── Chart sheet (standalone chart tab) ───────────────────────
            sh.Activate
            sh.ChartArea.Copy
        Else
            ' ── Regular worksheet (cells + embedded charts/images) ────────
            sh.Activate

            ' Expand selection to cover any objects that extend past UsedRange
            Dim rng As Range
            Set rng = ExpandRangeToObjects(sh)
            rng.CopyPicture Appearance:=xlScreen, Format:=xlPicture
        End If

        Application.ScreenUpdating = True

        ' ── Paste into the slide and fit to available area ───────────────
        PPTSlide.Shapes.Paste

        Dim pic As Object
        Set pic = PPTSlide.Shapes(PPTSlide.Shapes.Count)

        Dim availW As Single, availH As Single
        availW = PPTPres.PageSetup.SlideWidth  - 2 * MARGIN
        availH = PPTPres.PageSetup.SlideHeight - topOffset - MARGIN

        ' Scale uniformly to fill available space
        Dim scale As Single
        scale = WorksheetFunction.Min(availW / pic.Width, availH / pic.Height)

        pic.LockAspectRatio = msoTrue
        pic.Width  = pic.Width  * scale
        pic.Height = pic.Height * scale

        ' Centre horizontally; align to top of content area
        pic.Left = (PPTPres.PageSetup.SlideWidth - pic.Width) / 2
        pic.Top  = topOffset

        Application.CutCopyMode = False   ' clear clipboard

        Debug.Print "  Slide " & slideIdx & ": " & sheetName

ContinueLoop:
        Set sh  = Nothing
        Set pic = Nothing
    Next entry

    ' ── Save the presentation ─────────────────────────────────────────────
    If slideIdx = 0 Then
        MsgBox "No slides were created. Check the sheet names in SheetConfig().", _
               vbExclamation, "ExportToPowerPoint"
        GoTo Cleanup
    End If

    If OUTPUT_PATH = "" Then
        Dim baseName As String
        baseName = Left(ThisWorkbook.Name, InStrRev(ThisWorkbook.Name, ".") - 1)
        outPath = ThisWorkbook.Path & Application.PathSeparator & _
                  baseName & "_slides.pptx"
    Else
        outPath = OUTPUT_PATH
    End If

    PPTPres.SaveAs outPath, 24    ' 24 = ppSaveAsOpenXMLPresentation (.pptx)

    MsgBox slideIdx & " slide(s) created and saved to:" & vbLf & vbLf & _
           outPath, vbInformation, "ExportToPowerPoint"

Cleanup:
    Set PPTSlide = Nothing
    Set PPTPres  = Nothing
    Set PPTApp   = Nothing
    Exit Sub

End Sub


' ── Private helpers ───────────────────────────────────────────────────────

' Delete any placeholder shapes left over from the slide layout
Private Sub ClearPlaceholders(slide As Object)
    Dim i As Long
    For i = slide.Shapes.Count To 1 Step -1
        If slide.Shapes(i).Type = 14 Then   ' msoPlaceholder = 14
            slide.Shapes(i).Delete
        End If
    Next i
End Sub


' Add a solid title bar across the top of the slide
Private Sub AddTitleBar(slide As Object, title As String, _
                         slideWidth As Single, barHeight As Single)
    Dim bar As Object
    ' msoShapeRectangle = 1
    Set bar = slide.Shapes.AddShape(1, 0, 0, slideWidth, barHeight)
    bar.Line.Visible = msoFalse
    bar.Fill.ForeColor.RGB = RGB(30, 39, 97)        ' dark navy

    With bar.TextFrame2
        .TextRange.Text = title
        .TextRange.Font.Bold = msoTrue
        .TextRange.Font.Size = 16
        .TextRange.Font.Fill.ForeColor.RGB = RGB(255, 255, 255)
        .TextRange.ParagraphFormat.Alignment = 2    ' ppAlignCenter
        .VerticalAnchor = 3                         ' msoAnchorMiddle
        .MarginLeft  = 10
        .MarginRight = 10
    End With
End Sub


' Expand UsedRange to include any chart/image objects that extend beyond it.
' This ensures CopyPicture captures the full visible content of the sheet.
Private Function ExpandRangeToObjects(ws As Worksheet) As Range
    Dim rng As Range
    Set rng = ws.UsedRange

    Dim shp As Shape
    For Each shp In ws.Shapes
        ' Find the cells covered by this shape's bounding box
        Dim topCell    As Range, botCell As Range
        On Error Resume Next
        Set topCell = ws.Cells(shp.TopLeftCell.Row, shp.TopLeftCell.Column)
        Set botCell = ws.Cells(shp.BottomRightCell.Row, shp.BottomRightCell.Column)
        On Error GoTo 0

        If Not topCell Is Nothing And Not botCell Is Nothing Then
            Set rng = Union(rng, ws.Range(topCell, botCell))
        End If
    Next shp

    Set ExpandRangeToObjects = rng
End Function
