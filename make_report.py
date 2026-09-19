"""Build the 2-4 page PDF report from the notebook's saved outputs.

Run the notebook first (it writes metrics.json and figures/), then:
    python make_report.py
Requires: pip install reportlab
"""
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem,
                                Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

M = json.load(open("metrics.json"))
FIG = Path("figures")
OUT = "task1_ml_report.pdf"

te, tr, bl = M["linreg_test"], M["linreg_train"], M["baseline_test"]
cmp_ = M["comparison"]
hgb = cmp_["HistGradientBoosting"]
mae_cut = 1 - hgb["Test MAE"] / te["MAE"]

NAVY = colors.HexColor("#1f3a5f")
styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13.2, spaceAfter=4)
small = ParagraphStyle("small", parent=body, fontSize=8, leading=10.5, textColor=colors.HexColor("#555555"))
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=13, textColor=NAVY, spaceBefore=8, spaceAfter=4)
title = ParagraphStyle("title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=19, textColor=NAVY, spaceAfter=2, alignment=0)
sub = ParagraphStyle("sub", parent=body, fontSize=10.5, textColor=colors.HexColor("#444444"), spaceAfter=6)
cell = ParagraphStyle("cell", parent=body, fontSize=8.8, leading=11, spaceAfter=0)


def bullets(items):
    return ListFlowable([ListItem(Paragraph(t, body), leftIndent=10) for t in items],
                        bulletType="bullet", start="\u2022", leftIndent=12, bulletFontSize=8)


def img(name, width_cm):
    from PIL import Image as PILImage
    w, h = PILImage.open(FIG / name).size
    return Image(str(FIG / name), width=width_cm * cm, height=width_cm * cm * h / w)


hdr = ParagraphStyle("hdr", parent=body, fontName="Helvetica-Bold", fontSize=8.8, leading=11, textColor=colors.white, spaceAfter=0)


def table(data, col_widths, header=True, bold_row=None):
    data = [[Paragraph(c.replace("R2", "R<super>2</super>"), hdr) for c in data[0]]] + data[1:]
    t = Table(data, colWidths=[c * cm for c in col_widths], hAlign="LEFT")
    st = [("FONTNAME", (0, 0), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), 8.8),
          ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8cfd9")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
               ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    if bold_row is not None:
        st += [("FONTNAME", (0, bold_row), (-1, bold_row), "Helvetica-Bold"),
               ("BACKGROUND", (0, bold_row), (-1, bold_row), colors.HexColor("#e8eef7"))]
    t.setStyle(TableStyle(st))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(2 * cm, 1.1 * cm, "AI & ML Task 1: Linear Regression House Price Predictor")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


s = []
s += [Paragraph("House Price Prediction with Linear Regression", title),
      Paragraph("Maincrafts Technology | Artificial Intelligence &amp; Machine Learning, Task 1", sub)]

# 1 -------------------------------------------------------------------------
s += [Paragraph("1. Objective and data", h1),
      Paragraph("This project walks through a complete machine-learning workflow (load, explore, preprocess, train, "
                "evaluate, report) by predicting the <b>median house value</b> of California census block groups with a "
                "<font face='Courier'>LinearRegression</font> model from scikit-learn. The dataset is the built-in "
                "California Housing data: <b>20,640 rows, 8 numeric features</b> (median income, house age, average rooms, "
                "average bedrooms, population, average occupancy, latitude, longitude) and the target "
                "<font face='Courier'>MedHouseVal</font> in units of $100,000. There are no missing values and no duplicate rows. "
                f"The data was split 80/20 ({M['n_train']:,} training and {M['n_test']:,} test rows, "
                "<font face='Courier'>random_state=42</font>).", body)]

# 2 -------------------------------------------------------------------------
s += [Paragraph("2. Exploratory data analysis", h1),
      bullets([
          "<b>Skew and outliers.</b> Rooms, bedrooms, occupancy and population are heavily right-skewed. A few block groups "
          "average 140+ rooms or 1,200+ people per household, which stretches the histograms (Figure 1).",
          "<b>Capped values.</b> The target is capped at $500k (about 4.7% of rows sit exactly at the cap); house age (52) and "
          "income (15) also pile up at their maximum. A linear model cannot learn what happens beyond these caps.",
          "<b>Income dominates.</b> Median income has by far the strongest correlation with price (r = 0.69); every other "
          "feature is at about 0.15 or below (Figure 2).",
          "<b>Multicollinearity.</b> Latitude and longitude are strongly negatively correlated (r = -0.92) and so are average "
          "rooms and bedrooms (r = 0.85), which makes individual coefficients unstable.",
          "<b>Location is non-linear.</b> Expensive blocks hug the coast (Bay Area, Los Angeles, San Diego) while inland "
          "areas are cheap (Figure 3), a pattern a straight line in latitude/longitude cannot capture.",
      ]),
      Spacer(1, 4), img("fig1_distributions.png", 14.5),
      Paragraph("Figure 1. Distribution of every feature and the target.", small)]

fig2 = img("fig2_correlation.png", 7.4)
side = Paragraph("<b>Figure 2.</b> Correlation matrix. <font face='Courier'>MedInc</font> is the only feature with a strong "
                 "correlation with the target; the two problem pairs are latitude/longitude and rooms/bedrooms.", small)
s += [Spacer(1, 4), Table([[fig2, side]], colWidths=[8.6 * cm, 8.4 * cm], hAlign="LEFT",
                          style=TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")])),
      Spacer(1, 4), img("fig3_income_and_map.png", 15.0),
      Paragraph("Figure 3. Income vs house value (left) and house value by location (right).", small)]

# 3 -------------------------------------------------------------------------
s += [Paragraph("3. Model and results", h1),
      Paragraph("All 8 features were used as they are. Ordinary least squares is not affected by feature scale, so no scaling "
                "was applied. The model was trained on the training set and scored once on the untouched test set "
                "using MAE, RMSE and R<super>2</super>. A model that always predicts the training mean is the baseline.", body)]
rows = [["Model / split", "MAE", "RMSE", "R2"],
        ["Baseline (predict mean), test", f"{bl['MAE']:.3f}", f"{bl['RMSE']:.3f}", f"{bl['R2']:.3f}".replace("-0.000", "0.000")],
        ["Linear Regression, train", f"{tr['MAE']:.3f}", f"{tr['RMSE']:.3f}", f"{tr['R2']:.3f}"],
        ["Linear Regression, test", f"{te['MAE']:.3f}", f"{te['RMSE']:.3f}", f"{te['R2']:.3f}"]]
s += [KeepTogether([table(rows, [7.2, 2.6, 2.6, 2.6], bold_row=3),
                    Paragraph("Table 1. Error metrics (MAE and RMSE in units of $100,000).", small)]),
      Spacer(1, 4),
      bullets([
          f"The model explains about <b>{te['R2']:.0%}</b> of the variance in house prices and clearly beats the baseline "
          f"(R<super>2</super> of about 0). The typical error (MAE) is about <b>${te['MAE'] * 100_000 / 1000:,.0f}k</b>, "
          "compared with a median house value of about $180k.",
          f"Train and test scores are close (R<super>2</super> {tr['R2']:.3f} vs {te['R2']:.3f}), so the model is <b>not overfitting</b>. "
          "The limitation is <b>underfitting</b>: a straight-line model is too simple for this data.",
          "Median income has the expected positive coefficient; latitude and longitude are negative (a crude stand-in for the "
          "coastal effect). The bedroom and room coefficients have counter-intuitive signs because of multicollinearity and "
          "should not be over-interpreted.",
      ]),
      Spacer(1, 4),
      Table([[img("fig4_actual_vs_predicted.png", 5.0),
              Paragraph("<b>Figure 4.</b> Actual vs predicted on the test set. Predictions follow the red line only loosely and "
                        "the model under-predicts expensive houses; the horizontal stripe at 5.0 is the capped rows.", small)]],
            colWidths=[5.6 * cm, 11.4 * cm], hAlign="LEFT", style=TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")])),
      Spacer(1, 4), img("fig5_residuals.png", 12.8),
      Paragraph("Figure 5. Residuals vs predicted (left) and residual distribution (right). Residuals are centred on zero, but the "
                "spread widens for higher predictions and the diagonal band is the capped rows. A few predictions are negative, "
                "because a linear model has no bound on its output.", small)]

# 4 -------------------------------------------------------------------------
comp_rows = [["Model", "CV R2 (5-fold)", "Test MAE", "Test RMSE", "Test R2"]]
for name, v in cmp_.items():
    comp_rows.append([name, f"{v['CV R2 (mean)']:.3f}", f"{v['Test MAE']:.3f}", f"{v['Test RMSE']:.3f}", f"{v['Test R2']:.3f}"])
s += [Paragraph("4. Improvement ideas", h1),
      Paragraph("A quick experiment compared the baseline with two alternatives (5-fold cross-validation on the training set, "
                "plus test scores).", body),
      KeepTogether([table(comp_rows, [5.0, 3.0, 2.8, 2.8, 2.8], bold_row=3),
                    Paragraph("Table 2. Model comparison (MAE and RMSE in units of $100,000).", small)]),
      Spacer(1, 4),
      bullets([
          "<b>Ridge gives the same score as plain linear regression</b>, so regularisation does not help. The problem is model "
          "flexibility, not overfitting.",
          f"<b>Gradient boosting lifts R<super>2</super> from {te['R2']:.2f} to {hgb['Test R2']:.2f}</b> and cuts MAE by about "
          f"{mae_cut:.0%}, because it captures non-linear location effects and feature interactions.",
          "<b>Handle outliers, caps and skew:</b> clip extreme room and occupancy values, exclude the capped rows, and model "
          "log(price) to stabilise the residual spread.",
          "<b>Engineer features:</b> distance to the coast or major cities, rooms per person, bedroom ratio, latitude x longitude interactions.",
          "<b>Add non-linear terms and tune:</b> polynomial features on income and location, then cross-validated hyper-parameter search.",
      ]),
      KeepTogether([Paragraph("5. Conclusion", h1),
      Paragraph(f"A plain linear regression is an interpretable baseline (R<super>2</super> {te['R2']:.2f}, MAE about "
                f"${te['MAE'] * 100:,.0f}k). Median income is the main driver of price and location the second, but the "
                "location effect is non-linear, and the capped target and outliers add further error. Moving to a non-linear "
                "model and improving the features are the clearest routes to better predictions.", body)])]

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.7 * cm, bottomMargin=1.8 * cm,
                        title="House Price Prediction with Linear Regression", author="")
doc.build(s, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
