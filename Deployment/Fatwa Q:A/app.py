from flask import Flask, render_template , request 
import pandas as pd
import pickle
from farasa.stemmer import FarasaStemmer
from bertopic import BERTopic
from flair.embeddings import TransformerDocumentEmbeddings
import pyarabic.araby as ar
from camel_tools.utils import normalize as n
import re
from nltk.corpus import stopwords
app = Flask(__name__, template_folder='templates')
with BERTopic.load(open('my_model.h5','rb')) as model: 
  loaded_model = (model)

@app.route('/')
def home():
  return render_template("index.html")
@app.route('/',methods=['POST'])         
def predict():
  
  inp = str(request.form.get('question'))
  
  inp = ar.strip_tashkeel(inp)
  inp = re.sub("گ", "'ك", inp)
  inp = re.sub("ؤ", "ء", inp)
  inp = re.sub("ئ", "ء", inp)
  inp = re.sub('؟' , '', inp)
  inp = re.sub('،' , '', inp)
  inp = re.sub('/' , '', inp)
  inp = re.sub('-' , '', inp)
  inp = re.sub('!' , '', inp)
  inp = n.normalize_alef_ar(inp)
  inp = n.normalize_alef_maksura_ar(inp)
  inp = n.normalize_teh_marbuta_ar(inp)
  inp = inp.replace('اسوي','افعل')
  inp = inp.replace('نسوي','نفعل')
  inp = inp.replace('ابدا','')
  stop_words = stopwords.words('arabic')
  inp =  ' '.join([word for word in inp.split() if word not in (stop_words)])
  my_word = ['وماهي' ,'ايش' ,'شلون' ,'وش' ,'وين' ,'اين' ,'ماهي' ,'ازاي' ,'شنو' ,'عاوزه' ,'فين' ,'عايزه' ,'ازاي' ,'وشي' ,'ما' ,'هي' ,'او' ,'او' ,'وازاي' ,'بدي' ,'اذا' ,'هيه' ,'وشلون' ,'اش' ,'هوا' ,'ايه' ,'ماهو' 'هوا' ,'منين' ,'اني' ,'ان' ,'علي' ,'شي' , 'عندما', 'فما', 'شو','ماهيا' , 'انا', 'وشهي', 'وشكرا','اللي', 'الي' ]
  inp = ' '.join([word for word in inp.split() if word not in (my_word)])
  st = FarasaStemmer()
  inp = st.stem(inp)
  p = loaded_model.transform()
  if p[0] == [0]:
    result = ['أن وقته يجوز في يوم العيد واليوم الذي بعده حتى نهاية الشهر']
  elif p[0] ==[-1]:
    result = ['عيد صياغة السوال ارجوك']
  elif p[0] == [1]:
    result = ["نعم يوجد ركعتين تحية مسجد ، وإذا أتيت وطفت كفاك عن تحية المسجد وإلا تصلي ركعتين أو ما تيسر ، لكن لا يلزمك في كل دخول طواف ، لو طافت الناس كلها ما خلا المطاف ."]
  elif p[0] == [2]:
    result = ['ولهذا صلاة الفريضة لا تجوز فيه']
  elif p[0] == [3]:
    result = ["البدء بالإحرام، فالسعي، فالوقوف بعرفة، فالمبيت بمزدلفة، فرمي جمرة العقبة، فالتحلل من الإحرام، فطواف الإفاضة، فرمي الجمرات، ثم طواف الوداع."]
  elif p[0] == [4]:
    result = ["إذا دفع الحجاج من عرفة إلى المزدلفة ووصلوا إليها فإنه يشرع لهم الآذان"]
  elif p[0] == [5]:
    result = [" الفِسق والجدال، الطِّيب، قلم الأظافر وإزالة شَعر الرأس، الجماع ودواعيه، الصَّيد، قَطع شجر وحشيش الحَرَم، إما للرجل: لبس المخيط، تغطية الرأس كله أو بعضه. بالنسبة للنساء: تغطية الوجه وكفيها."]
  elif p[0] == [6]:
    result = [" لاأرى مانع من ذلك إذا كان الشخص متوفي "]
  elif p[0] == [7]:
    result = ["لأهل المدينة ذي الخليفة ، ولأهل الشام ومصر وشمال أفريقيا الجحفة ، ولأهل اليمن والساحل يلملم. لا يجوز تجاوز الميقات بدون إحرام "]
 
  return render_template('index.html',val=result)
#result = predict(str(request.form.get('question')))

    









if __name__=="__main__":
    app.run(debug=True)
