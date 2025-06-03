import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, Entry
from PIL import Image, ImageTk
import numpy as np
import smtplib
import ssl
from email.message import EmailMessage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
#### خليت تشات يساعدني في هل gui كونه مش كثير مهتم اتعلم اعمل gui
# تحميل النموذج
model = load_model("model.h5")
class_names = ["Infarction", "Ischemia", "Normal"]
IMG_SIZE = (224, 224)
last_image_path = None

# التنبؤ بالصورة
def predict_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

# تحميل الصورة
def upload_image():
    global last_image_path
    file_path = filedialog.askopenfilename(
        title="اختر صورة",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        return

    last_image_path = file_path

    # عرض الصورة
    img = Image.open(file_path).resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    # التنبؤ
    predicted_class, confidence = predict_image(file_path)
    result_text = f"🔍 النتيجة: {predicted_class}\n🎯 الدقة: {confidence * 100:.2f}%"

    if confidence < 0.7:
        result_text += "\n⚠️ الرجاء مراجعة الطبيب لتأكيد النتيجة"
        result_label.config(fg="#c62828")
        show_email_fields()
    else:
        result_label.config(fg="#01579b")
        hide_email_fields()

    result_label.config(text=result_text)

# إعادة تعيين
def reset_gui():
    global last_image_path
    last_image_path = None
    image_label.config(image="", text="🖼️ لم يتم تحميل صورة", font=("Segoe UI", 14, "italic"))
    image_label.image = None
    result_label.config(text="")
    email_entry.delete(0, tk.END)
    hide_email_fields()

# إرسال الإيميل

def send_email():
    recipient = email_entry.get()
    if not recipient or not last_image_path:
        result_label.config(text="❗ يرجى إدخال الإيميل وتحميل صورة أولًا", fg="#d32f2f")
        return

    try:
        email_sender = "YourEmail@---.com"
        email_password = "ur application email password "

        msg = EmailMessage()
        msg["Subject"] = "🔍 نتيجة فحص القلب باستخدام الذكاء الاصطناعي"
        msg["From"] = email_sender
        msg["To"] = recipient

        # استخراج النص مع حذف تحذير الواجهة وإضافة طلب إعادة التصنيف
        result_text = result_label["text"].replace("⚠️ الرجاء مراجعة الطبيب لتأكيد النتيجة", "").strip()
        email_body = f"""
تم تنفيذ تشخيص أولي باستخدام النظام الذكي، وكانت النتيجة كما يلي:

{result_text}

🌀 الرجاء إعادة تصنيف الصورة من قبل الطبيب المختص للتحقق من دقة التشخيص الآلي.

مرفقة صورة القلب التي تم تحليلها.
"""

        msg.set_content(email_body)

        with open(last_image_path, "rb") as f:
            file_data = f.read()
            file_name = os.path.basename(last_image_path)

        msg.add_attachment(file_data, maintype="image", subtype="jpeg", filename=file_name)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
            smtp.login(email_sender, email_password)
            smtp.send_message(msg)

        result_label.config(text="📤 تم إرسال النتيجة والصورة إلى البريد الإلكتروني", fg="#2e7d32")
    except Exception as e:
        result_label.config(text=f"❌ فشل الإرسال: {e}", fg="#c62828")


# إظهار حقول الإيميل
def show_email_fields():
    email_label.pack(pady=(10, 0))
    email_entry.pack(pady=5)
    send_btn.pack(pady=20)

# إخفاء حقول الإيميل
def hide_email_fields():
    email_label.pack_forget()
    email_entry.pack_forget()
    send_btn.pack_forget()

# إعداد الواجهة
background_color = "#e0f7fa"
root = tk.Tk()
root.title("🫀 نظام ذكي لتشخيص الحالات القلبية")
root.geometry("850x720")
root.configure(bg=background_color)

title_label = Label(root, text="نظام ذكي لتشخيص الحالات القلبية", font=("Segoe UI", 22, "bold"), bg=background_color, fg="#004d40")
title_label.pack(pady=20)

content_frame = Frame(root, bg=background_color)
content_frame.pack()

upload_btn = Button(content_frame, text="📂 تحميل صورة", command=upload_image, font=("Segoe UI", 14, "bold"),
                    bg="#00acc1", fg="white", activebackground="#4dd0e1", padx=20, pady=10, bd=0, cursor="hand2")
upload_btn.grid(row=0, column=0, padx=10, pady=10)

reset_btn = Button(content_frame, text="🔄 إعادة تعيين", command=reset_gui, font=("Segoe UI", 14, "bold"),
                   bg="#007c91", fg="white", activebackground="#4dd0e1", padx=20, pady=10, bd=0, cursor="hand2")
reset_btn.grid(row=0, column=1, padx=10, pady=10)

image_label = Label(content_frame, bg=background_color, width=300, height=300, text="🖼️ لم يتم تحميل صورة",
                    font=("Segoe UI", 14, "italic"), fg="#37474f")
image_label.grid(row=1, column=0, columnspan=2, pady=20)

result_label = Label(content_frame, text="", font=("Segoe UI", 16, "bold"), bg=background_color, fg="#004d40")
result_label.grid(row=2, column=0, columnspan=2, pady=20)

# إدخال الإيميل (مخفي بالبداية)
email_label = Label(root, text="📧 بريد الطبيب:", font=("Segoe UI", 14), bg=background_color, fg="#004d40")
email_entry = Entry(root, font=("Segoe UI", 14), width=30, justify="center")
send_btn = Button(root, text="📩 إرسال النتيجة على الإيميل", command=send_email, font=("Segoe UI", 14, "bold"),
                  bg="#009688", fg="white", activebackground="#4db6ac", padx=30, pady=10, bd=0, cursor="hand2")

# بدء الواجهة
hide_email_fields()
root.mainloop()
