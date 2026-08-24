# Nabd Agent for Termux

**Nabd** هو وكيل برمجة يعمل من سطر الأوامر داخل مجلد المشروع. يقرأ بنية المشروع وملفاته، يضع خطة، يقترح تعديلات كاملة للملفات، ينفّذ الاختبارات أو أوامر التحقق، ثم يعيد محاولة الإصلاح عند ظهور أخطاء. صُمّم الإصدار الأول دون حزم Python خارجية، لذلك يمكن تشغيله في Termux بعد تثبيت Python فقط.

## التصميم

يستخدم الوكيل آلة الحالات التي شاركتها في ملف `fsm.py`، مع الحفاظ على المبدأ الأساسي: لا يمكن الانتقال من التخطيط إلى الاكتمال مباشرة، ولا يُعلن نجاح المهمة قبل مرحلة التحقق.

| المرحلة | الوظيفة |
| --- | --- |
| `PLANNING` | يرسل المهمة وبنية المشروع إلى نموذج الذكاء الاصطناعي للحصول على خطة JSON. |
| `EXECUTING` | يقرأ الملفات أو يكتبها أو ينفذ الأوامر المطلوبة داخل جذر المشروع. |
| `VERIFYING` | يشغّل أوامر التحقق التي اقترحها النموذج، مثل اختبارات المشروع أو فحص الترجمة. |
| `COMPLETED` | جميع أوامر التحقق نجحت. |
| `REJECTED` | رفض المستخدم إجراءً أو فشلت محاولات الإصلاح المحدودة. |

> **مبدأ الأمان:** لا يستطيع الوكيل استخدام مسار خارج جذر المشروع، ولا يستطيع قراءة `.env` أو مفاتيح SSH أو مجلد `.git`. يعمل Nabd تلقائيًا افتراضيًا وفق طلبك، مع بقاء فحوص Jail إلزامية؛ استخدم `--confirm` للموافقة التفاعلية عند الحاجة.

## التثبيت في Termux

ثبّت Termux من مصدر موثوق، ثم نفّذ الأوامر التالية داخل Termux بعد نقل مجلد المشروع إلى الهاتف أو استنساخه من مستودعك:

```sh
pkg update -y
pkg install -y python git
cd /path/to/nabd-agent-termux
bash install.sh
```

إذا كان المشروع داخل ذاكرة الهاتف، امنح Termux صلاحية الوصول مرة واحدة ثم استخدم المسار الذي يظهره الأمر التالي:

```sh
termux-setup-storage
cd ~/storage/shared/nabd-agent-termux
bash install.sh
```

لا يحتاج Nabd إلى `pip install` في هذا الإصدار؛ فهو يستخدم مكتبات Python القياسية فقط.

## إعداد مزود الذكاء الاصطناعي

انسخ نموذج الإعدادات ثم ضع مفتاح مزود واحد على الأقل في جلسة Termux الحالية:

```sh
cp .env.example .env.local
export OPENAI_API_KEY="ضع_المفتاح_هنا"
# أو استخدم Gemini:
# export GEMINI_API_KEY="ضع_المفتاح_هنا"
# أو استخدم NVIDIA NIM:
# export NVIDIA_API_KEY="nvapi-ضع-المفتاح-هنا"
```

يمكن بدلًا من ذلك تحميل متغيرات بيئة من ملف خاص لا تضفه إلى Git:

```sh
set -a
. ./.env.local
set +a
```

يدعم الوكيل `--provider auto` تلقائيًا؛ عند وجود `NVIDIA_API_KEY` يختار NVIDIA أولًا، ثم OpenAI، ثم Gemini. ويمكنك تحديد المزود صراحةً باستخدام `--provider openai` أو `--provider gemini` أو `--provider nvidia`. لا تضع مفتاح NVIDIA في GitHub، بل احفظه داخل `.env.local` المحلي فقط.

```sh
export NABD_PROVIDER=openai
# أو
export NABD_PROVIDER=gemini
# أو NVIDIA NIM
# export NABD_PROVIDER=nvidia
# export NABD_NVIDIA_MODEL="meta/llama-3.1-8b-instruct"
# export NABD_NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1/chat/completions"
```

## الاستخدام

شغّل Nabd من جذر المشروع المطلوب:

```sh
nabd "حل مشكلة تسجيل الدخول وإضافة اختبارات لها"
```

أمثلة أخرى:

```sh
nabd "راجع هذا المشروع، أصلح أخطاء Python، وشغّل الاختبارات" --root ~/projects/app
nabd "أضف README مختصرًا وملف إعدادات آمنًا" --provider gemini
nabd "نفّذ الإصلاحات والاختبارات دون طلب موافقة لكل خطوة" --auto
```

يعمل Nabd الآن بالتنفيذ التلقائي افتراضيًا، أي إنه لا يطلب موافقة قبل الكتابة أو تشغيل الأوامر. استخدم `--confirm` إذا أردت إعادة الموافقة التفاعلية، بينما يبقى `--yes` و`--auto` متاحين كأسماء صريحة للوضع التلقائي. هذا السلوك مقصود لأنه متطلب المستخدم؛ لذلك لا يُطبّق اختبار CLI المشارك الذي يفرض `auto_approve=False` افتراضيًا. استخدم التنفيذ التلقائي فقط داخل مجلد مشروع تثق به؛ إذ إن Jail ما زال يحظر أنماطًا خطرة محددة.
 يحد الخيار `--max-rounds` عدد دورات الإصلاح، وقيمته الافتراضية خمس دورات. بعد التحقق الناجح يُحفظ سجل الأدلة في `.nabd/evidence.json`، ولا تُعد المهمة مكتملة إذا لم يوجد أمر تحقق ناجح أو إذا تغيّر هاش ملف مسجل.

```sh
nabd "أصلح جميع الاختبارات الفاشلة" --max-rounds 3
```

## سجل الأدلة

يستخدم الوكيل `EvidenceStore v2`. الأدلة من نوع `OBSERVED` تُسجل فقط بعد تحقق فعلي: وجود الملف على القرص مع بصمة SHA-256، أو نجاح أمر تحقق بخروج يساوي صفرًا. أما المعلومات التي ينتجها النموذج فتظل توجيهًا تخطيطيًا ولا تكفي وحدها لإعلان النجاح. يعيد `is_usable_for_completion()` التحقق من الأدلة الحالية، بينما لا تمنع محاولات فاشلة سابقة إصلاحًا لاحقًا داخل نفس المهمة إذا ظهرت أدلة جديدة صالحة. يُحفظ السجل محليًا في `.nabd/evidence.json` وهو متجاهل في Git افتراضيًا.

## تشغيل الاختبارات المحلية

يمكن اختبار مكوّن FSM دون مفتاح API:

```sh
python3 -m unittest discover -s tests -v
python3 -m compileall -q nabd
```

## الملفات الرئيسية

| الملف | الغرض |
| --- | --- |
| `nabd/fsm.py` | آلة الحالات المستوحاة من الملف المشارك. |
| `nabd/agent.py` | حلقة التخطيط والتنفيذ والتحقق والإصلاح، وتطهير بادئات `run_command` الوهمية قبل تنفيذ التحقق. |
| `nabd/llm.py` | اتصال OpenAI وGemini باستخدام HTTP ومكتبات Python القياسية. |
| `nabd/tools.py` | طبقة تحويل ToolCall إلى ToolResult يحمل RawFacts، دون إصدار Evidence من الأداة. |
| `nabd/write_tool.py` | أداة كتابة تعيد RawFacts عن الملف والنسخة الاحتياطية؛ لا تصدر Evidence. |
| `nabd/read_tool.py` | أداة قراءة تعيد RawFacts بالمحتوى والهاش وبيانات truncation. |
| `nabd/list_tool.py` | أداة سرد تعيد files/count/truncated داخل RawFacts بحد 200 ملف، وتتجاهل `.nabd` وآثار التشغيل. |
| `nabd/search_tool.py` | بحث يعيد MATCH/NO_MATCH/TOOL_ERROR وبيان fallback داخل RawFacts. |
| `nabd/shell_tool.py` | أداة shell تعيد exit_code/timeout/stdout/stderr/signal داخل RawFacts. |
| `nabd/jail.py` | عزل مسارات workspace وفحص الروابط الرمزية وأنماط الأوامر الخطرة. |
| `nabd/evidence.py` | EvidenceStore v2؛ يعيد قراءة الملفات ويتحقق من RawFacts وtask_id وfresh/relevant/valid قبل إصدار `OBSERVED`، ولا يجعل محاولات فاشلة سابقة تمنع إصلاحًا موثقًا لاحقًا. |
| `nabd/tools.py` | ToolExecutor؛ يحتوي ToolError وJailError وأخطاء الملفات ويحوّلها إلى ToolResult فاشل بدل إسقاط جلسة الوكيل. |
| `nabd/cli.py` | واجهة سطر الأوامر. |
| `install.sh` | تثبيت الأمر `nabd` وربطه داخل `$PREFIX/bin`. |
| `tests/test_fsm.py` | اختبارات آلة الحالات. |
| `tests/test_jail.py` | اختبارات عزل المسارات، الروابط الرمزية، اجتياز المسارات، الأوامر الخطرة، واحتواء قراءة ملف مفقود. |
| `tests/test_tool_evidence_integration.py` | اختبار التكامل والفصل بين RawFacts وEvidence، مع task_id وكشف العبث. |

## ملاحظات تشغيلية

لا ترسل مفاتيح API داخل وصف المهمة أو ملفات المشروع. يرسل الوكيل إلى المزود قائمة الملفات غير الحساسة ونتائج الأدوات، بينما تحجب طبقة الأدوات ملفات الأسرار ومجلد Git. التنفيذ التلقائي هو الوضع الافتراضي، ويمكن استخدام `--confirm` للمهام التي تتطلب موافقة تفاعلية.

النسخة الحالية مناسبة لوكيل محلي فردي. ويمكن توسيعها لاحقًا بإضافة دعم Git diff قبل وبعد التعديل، وتخزين جلسات العمل، وتكامل مع MCP أو أدوات lint خاصة بكل لغة، من دون تغيير آلة الحالات الأساسية.

## المرجع

[1]: https://www.meta.ai/share/a/669def02-a8cd-4445-a728-720dd60228dd "ملف fsm.py المشارك من المستخدم"

## حماية مساحة العمل

تتضمن النسخة الحالية `nabd/jail.py` كطبقة دفاعية إضافية مستوحاة من المشاركة الثالثة. تمنع أدوات الملفات المسارات خارج جذر المشروع، وتحجب أنماط `.git` و`.env` و`.ssh` و`.aws` و`.config` ومجلدات النظام. كما تفحص أوامر shell قبل تشغيلها وتحجب أنماطًا معروفة الخطورة مثل `rm -rf` على مسارات حساسة، و`sudo`، و`eval`، و`exec`، وتمرير تنزيلات `curl` أو `wget` إلى shell، والكتابة إلى `/etc` أو `/proc` أو `/sys`، واجتياز المسارات باستخدام `/` أو `\\`. وتتحقق من الروابط الرمزية حتى لا تؤدي إلى خارج مساحة العمل.

هذه الحماية لا تلغي ضرورة تشغيل الوكيل داخل مجلد تثق به. حتى مع التنفيذ التلقائي، لا تُعد Jail عزلًا كاملًا للنظام؛ فهي طبقة فحص لأنماط معروفة وليست بديلًا عن صلاحيات نظام التشغيل أو الحاويات.

[2]: https://www.meta.ai/share/a/d5503b58-4b22-4fab-ab28-1d45f23944c1 "ملف evidence.py المشارك من المستخدم"
[3]: https://www.meta.ai/share/a/54b39bd1-2c62-40e5-9533-f966e527998c "ملف jail.py المشارك من المستخدم"
[4]: https://www.meta.ai/share/a/7ed901ad-b86b-4e12-86d6-44e0c47f484a "اختبار Jail الشامل المشارك من المستخدم"
[5]: https://www.meta.ai/share/a/1f538f54-3766-4654-bc8a-4a71630fb00f "اختبار EvidenceStore الشامل المشارك من المستخدم"
[6]: https://www.meta.ai/share/a/d41ecede-7b6a-454d-bc08-912bed8bba4f "اختبار أمان CLI المشارك من المستخدم"
[7]: https://www.meta.ai/share/a/31e609ff-8bdd-4a62-8a5f-1d2d8e4be730 "مثال main.py المشارك من المستخدم"
[8]: https://www.meta.ai/share/a/60bb9db9-77f8-4adb-80cb-9bfe8ffebbe1 "README المرجعي المشارك من المستخدم"

## مراجعة README المرجعي

المشاركة [8] تصف نسخة أولية تحتوي على 4 ملفات أساسية و5 أدوات وتذكر 15 اختبارًا. النسخة الحالية أوسع من ذلك، وتحتوي على واجهة CLI، وأدوات `write/list/search/shell` المستقلة، و`nabd-verify`، وتغطية اختبار محدثة. لذلك استُخدمت المشاركة كمرجع تاريخي ولم تُستبدل بها وثائق المشروع الحالية.

## مراجعة مثال `main.py`

المشاركة [7] تقدم مثالًا تعليميًا صغيرًا يكتب ملفًا مؤقتًا ثم ينتقل في FSM إلى `COMPLETED`. لم تُنسخ كما هي لأن مسارات الاستيراد (`agent.core.*`) وواجهات `EvidenceStore` و`WriteTool` تختلف عن Nabd، كما أن المثال لا يشغّل أمر تحقق فعليًا قبل إعلان الاكتمال. التكامل الصحيح موجود في `nabd/agent.py` و`nabd/write_tool.py`، حيث تُسجّل الأدلة وتُعاد البصمات وتُشغّل أوامر التحقق قبل الانتقال إلى `COMPLETED`.

## تكامل الأدوات والأدلة

تستخدم الأدوات الخمس الفعلية `WriteTool` و`ReadTool` و`ListTool` و`SearchTool` و`ShellTool` نسخة `WorkspaceJail` الخاصة بمساحة العمل، لكنها تعيد `RawFacts` فقط ولا تصدر `OBSERVED`. تتضمن الحقائق الخام path وexists وsize وsha256 وmtime أو exit_code وtimeout وstdout وstderr وsignal وtruncated.

`EvidenceStore` هو الجهة الوحيدة التي تصدر `OBSERVED`: يعيد قراءة الملف من القرص، يعيد حساب SHA-256، ويتحقق من `task_id` و`relevant` و`fresh` و`valid`. نجاح shell أو وجود match في search لا يعني إكمال المهمة؛ بل يجب أن تكون الأدلة قابلة للاستخدام، ثم يسمح FSM بالانتقال من `VERIFYING` إلى `COMPLETED`. أما `ToolExecutor` فيحوّل RawFacts إلى نتائج العرض ويمررها إلى الوكيل دون إصدار دليل بنفسه.

يستخدم `SearchTool` ripgrep عند توفره، ويمرر الاستعلام كوسيط منفصل مع `--fixed-strings` لتجنب حقن shell؛ وإذا لم يتوفر ripgrep يستخدم fallback Python آمنًا داخل حدود Jail. وتوضح RawFacts الفرق بين `MATCH` و`NO_MATCH` و`TOOL_ERROR` و`FALLBACK_USED`.

## أداة التحقق المستقلة

يمكن استخدام الأمر `nabd-verify` لفحص ملف أو إنشاء نسخة احتياطية داخل مساحة العمل:

```sh
nabd-verify --root . hash src/main.py
nabd-verify --root . verify src/main.py SHA256_HASH
nabd-verify --root . jail src/main.py
nabd-verify --root . backup src/main.py
```

يُرجع أمر `verify` النص `OBSERVED` فقط عند تطابق البصمة، ويُرجع رمز خروج غير ناجح عند عدم التطابق. أما `backup` فينشئ نسخة مؤرخة داخل `.nabd/backups` ما لم تحدد مجلدًا آخر داخل جذر المشروع.

## verifier Rust الاختياري

تحتوي مجلدات `rust-verifier/` على `Cargo.toml` و`src/main.rs` من المشاركة الخامسة. هذه نسخة ثنائية اختيارية من أداة التحقق، وتحتاج إلى تثبيت Rust وCargo في Termux؛ أما Nabd الرئيسي و`nabd-verify` فيعملان بPython القياسي ولا يحتاجان إلى Rust.

لبناء النسخة الاختيارية وتشغيلها:

```sh
pkg install -y rust
cd rust-verifier
cargo build --release
./target/release/verifier --help
```

المسار الموصى به لمعظم مستخدمي Termux هو `nabd-verify`، لأن تثبيت Rust ووقت بناء الاعتماديات غير ضروريين للوظائف نفسها. أُبقيت نسخة Rust داخل المشروع لمن يريد binary مستقلًا أو تكاملًا لاحقًا مع نظام أكبر.
