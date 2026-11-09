<div dir="rtl">

# كشف الأجسام الجوية وتتبعها من فيديو الطائرة بدون طيار

مشروع تخرج في الرؤية الحاسوبية يدرس كشف الأجسام الطائرة وتتبعها في الفيديو. يركز المشروع على الأجسام الموجودة في الجو، وعلى استمرار التعرف على الجسم نفسه بين الإطارات، مع عرض نتائج يسهل فهمها.

افتح **`run.bat`** واختر محاكاة البث وفقد الاتصال، أو تسجيل مقاطع دورية، أو تحليل مقطع محفوظ. يشغّل مسار البث مقطعًا مسجلًا من الجهاز، ويحفظ فترة الانقطاع في مجلد يمثّل تخزين الـUAV، ثم ينقلها إلى مجلد المحطة ويكمل التحليل. نتائج التحليل تتضمن فيديو عليه مربعات التتبع وملف CSV وملخصًا قابلًا للطباعة. [التشغيل والنقل إلى لابتوب آخر](docs/session_workflow.md)

<div dir="ltr">

## Automatic recorded-video experiment

[`automation.py`](automation.py) receives one local video and automatically
calibrates the available model, analysis size and analysis stride. No model or
image-size selection is required from the user. After processing, it opens the annotated video in the default player. It writes its measurements,
selection reason, supported/missing classes, annotated video and CSV outputs.
This is the main automatic offline entry point. Separate manual and SAHI
baselines are retained for controlled comparisons.

```powershell
.\venv\Scripts\python.exe automation.py test01.mp4
```

[Automation method and limits](docs/automation.md) ·
[Four-class dataset reference](docs/dataset_sources.md) · [Verification](docs/automation_verification.md) · [Code map](docs/code_review.md)

The selection policy measures runtime and uses documented heuristics. It does
not establish best accuracy, train models, learn by itself or switch models
within a running video. Standard local weights currently cover airplane and
bird; drone and helicopter require compatible trained weights. Analysis-stride
changes are logged, and skipped frames are marked rather than counted as
negative detections.

</div>

## فكرة النظام والتقنيات

يمر الفيديو في التصميم بالمراحل التالية:

**كاميرا الطائرة ← استقبال الفيديو وقراءة الإطارات ← إعداد التحليل ← كشف الأجسام ← تتبعها ← عرض النتائج وحفظ الملفات.**

[مخطط سير العمل](docs/figures/System_Flowchart.svg) · [حالة الاستخدام](docs/figures/Use_Case.svg) · [تدفق البيانات](docs/figures/Data_Flow.svg)

- **YOLO:** يحدد فئة الجسم وموقعه داخل الصورة ودرجة الثقة في الكشف.
- **ByteTrack:** يربط الاكتشافات بين الإطارات ويعطي الجسم المتتبع رقم هوية.
- **SAHI:** يقسم الصورة إلى مناطق لإجراء الكشف عليها؛ وهو موجود في تجربة مستقلة داخل المستودع.
- **OpenCV:** يقرأ الفيديو ويعرض الإطارات المعلّمة ويحفظ الفيديو الناتج.

يسند التصميم اختيار الموديل وحجم إطار التحليل ومعدل المعالجة والطريقة والإعدادات إلى النظام. المستخدم يشاهد المخرجات، ولا يتولى تحليل الفيديو أو إعداد الذكاء الاصطناعي في هذا التصميم.

**حالة الكود:** تبدأ من `run.bat`، ويدير `uav_session.py` الجلسة. يستدعي `automation.py` لقياس خيارات الموديل وحجم التحليل على عينة من الفيديو، ثم يضبط معدل تحليل الإطارات أثناء المعالجة. مسار الكشف اليدوي مستقل للتجارب ذات الإعدادات الثابتة.

## مسار المقارنة اليدوي

ملف التشغيل اليدوي هو [`detect_airborne_targets.py`](detect_airborne_targets.py). يقرأ الإطارات بالتتابع، ويطبق الكشف والتتبع، ويعرض الفيديو أثناء المعالجة ما لم يُستخدم `--no-show`.

| العنصر | الوصف |
| --- | --- |
| مصدر الفيديو | مسار ملف أو رقم كاميرا يحدده `--source` في النسخة الحالية. |
| أوزان النموذج | ملف يحدده `--weights`. الفئات المدعومة تعتمد على تدريب هذه الأوزان. |
| حجم التحليل | قيمة `--imgsz`، أو `auto` لتجربة عدة أحجام على عينات من فيديو مسجل. |
| المتتبع | ByteTrack افتراضيًا. |
| الفيديو الناتج | `airborne_tracking.mp4` داخل مجلد المخرجات. |
| سجل الكشف | `airborne_detections.csv`: الإطار، الوقت، الفئة، الثقة، رقم التتبع ومربع الجسم داخل الصورة. |

خيار `--imgsz auto` يستخدم عدد الاكتشافات والثقة لتقدير حجم مناسب؛ وهو لا يقيس الدقة ولا يختار موديل YOLO تلقائيًا. إحداثيات المربع تخص الصورة، ولا تمثل موقعًا جغرافيًا أو ارتفاعًا حقيقيًا. وتتكرر أسماء ملفات المخرجات، لذلك يستخدم مجلد مستقل لكل تجربة.

## ملفات المشروع

| الملف | الوظيفة |
| --- | --- |
| [`uav_session.py`](uav_session.py) | محاكاة البث والانقطاع، التسجيل الدوري، وتحليل مقطع محفوظ. |
| [`automation.py`](automation.py) | Recorded-video automatic setup, adaptive analysis rate, tracking and result playback. |
| [`detect_airborne_targets.py`](detect_airborne_targets.py) | معالجة الفيديو، والكشف والتتبع، والعرض، وحفظ MP4 وCSV. |
| [`detect_video_sahi.py`](detect_video_sahi.py) | كشف باستخدام SAHI؛ لا يحتوي على تتبع ByteTrack. |
| [`prepare_dataset.py`](prepare_dataset.py) | فحص الصور وملصقات YOLO، وحساب التوزيع، وفصل المقاطع والمكررات بين مجموعات البيانات، وإعداد نسخة بيانات عند الطلب. |
| [`train_detector.py`](train_detector.py) | كود تدريب وتقييم يتطلب بيانات معلّمة ومتوافقة. |
| [`compare_models.py`](compare_models.py) | مقارنة نماذج على ملف بيانات يُحدَّد صراحةً؛ يجب مراجعة فئاته قبل التدريب، ولا تُختار بيانات تلقائيًا. |

## البيانات والتقييم

يعتمد التدريب على صور ذات فئات ومربعات صحيحة، ويحتاج تقييم التتبع أيضًا إلى هويات للأجسام عبر الإطارات. يجب إبقاء إطارات المقطع الواحد في مجموعة واحدة، وفحص الصور السلبية والمكررة قبل التدريب. حفظ التنبؤات وحده لا ينتج ملصقات صحيحة للتدريب.

أداة التحضير تقرأ ملف CSV يحتوي `image,label,sequence,split`. يمكنها الحفاظ على تقسيم مُراجع أو إنشاء تقسيم على مستوى المقاطع، وتنتج ملخصًا بالأعداد. لا تشغّل تدريبًا ولا تضيف ملصقات آلية. تفاصيل استخدامها في [دليل تحضير البيانات](docs/data_preparation.md).

- [Halmstad](https://github.com/DroneDetectionThesis/Drone-detection-dataset): مصدر لعدة فئات جوية. أعداد التقرير تخص مقاطع الفيديو المنشورة، وليست نتائج تجارب للمشروع.
- [MOT-FLY](https://github.com/CZC-123/MOT-FLY): بيانات تصوير جوي لأجسام UAV مع هويات للتتبع.
- [مثال كاميرا Gazebo الرسمي](https://github.com/gazebosim/gz-sim/blob/gz-sim8/examples/worlds/camera_sensor.sdf): مرجع مشهد الكاميرا المستخدم لفحص بيئة المحاكاة؛ لا يمثل اختبار دقة للكشف.

## الدراسات المرجعية

| الدراسة | موضوعها |
| --- | --- |
| [Elshaar et al., 2024](https://arxiv.org/abs/2410.00285) | مقارنة YOLOv5 وYOLOv8 مع ByteTrack وBoT-SORT. |
| [GL-DT، نسخة 2025](https://arxiv.org/abs/2510.09092v1) | دمج الكشف العام والمحلي والتتبع باستخدام JPTrack. |
| [UAVSwarm، 2022](https://doi.org/10.3390/rs14112601) | بيانات وتقييم لكشف وتتبع عدة أجسام UAV. |
| [Synthetic Fog، 2026](https://doi.org/10.3390/jimaging12090432) | أثر الضباب الاصطناعي والتدريب عليه في الكشف والتتبع. |

هذه أربع دراسات سابقة؛ الصف الخاص بمشروعنا في التقرير هو صف المقارنة معها. لا تُعد نتائجها دليلًا على دقة النموذج الحالي أو تفوقه.

[تقرير المشروع](https://docs.google.com/document/d/19CcMctNZK8znEe7cA6AAcPFP5l49S0wZL98S5AhCRVk/edit) · [تعليمات التشغيل](docs/running.md)

</div>
