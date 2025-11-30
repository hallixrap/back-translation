"""
Back Translation Project - Document Sources & Sample Materials
Stanford Clinical Translation Evaluation Framework

This script contains patient education documents across multiple categories
sourced from NIH/MedlinePlus and other official health education resources.

SOURCES VERIFIED: November 2025
- MedlinePlus Medical Encyclopedia (medlineplus.gov)
- NIH National Heart, Lung, and Blood Institute (nhlbi.nih.gov)
- NIH National Institute of Diabetes and Digestive and Kidney Diseases (niddk.nih.gov)
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from config import DOCUMENTS_DIR, DOCUMENT_CATEGORIES, logger

# =============================================================================
# DOCUMENT DATA STRUCTURE
# =============================================================================

@dataclass
class PatientDocument:
    """Represents a patient education document for translation."""
    doc_id: str
    title: str
    category: str
    topic: str
    source: str
    source_url: Optional[str]
    english_text: str
    word_count: int
    reading_level: Optional[str] = None
    professional_translation: Optional[dict] = None  # {language_code: text}

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


# =============================================================================
# SAMPLE DOCUMENTS - MULTI-CATEGORY PATIENT EDUCATION MATERIALS
# =============================================================================

SAMPLE_DOCUMENTS = [
    # =========================================================================
    # CARDIOLOGY - Verified from MedlinePlus Medical Encyclopedia
    # =========================================================================
    PatientDocument(
        doc_id="CARD_001",
        title="Heart Failure - Discharge",
        category="cardiology",
        topic="heart_failure",
        source="MedlinePlus Medical Encyclopedia",
        source_url="https://medlineplus.gov/ency/patientinstructions/000114.htm",
        english_text="""Heart Failure - Discharge

Heart failure is a condition in which the heart is no longer able to pump oxygen-rich blood to the rest of the body efficiently. When symptoms become severe, a hospital stay may be necessary. This article discusses what you need to do to take care of yourself when you leave the hospital.

When You're in the Hospital
You were in the hospital to have your heart failure treated. Heart failure occurs when the muscles of your heart are weak or have trouble relaxing, or both.

Your heart is a pump that moves fluids through your body. As with any pump, if the flow out of the pump is not enough, fluids do not move well and they get stuck in places they should not be. In your body, this means that fluid collects in your lungs, abdomen, and legs.

While you were in the hospital:
Your health care team closely adjusted the fluids you drank or received through an intravenous (IV) line. They also watched and measured how much urine you produced.
You may have received medicines to help your body get rid of extra fluids.
You may have had tests to check how well your heart was working.

What to Expect at Home
Your energy will slowly return. You may need help taking care of yourself when you first get home. You may feel sad or depressed. All of these things are normal.

Checking Yourself at Home
Weigh yourself every morning on the same scale when you get up -- before you eat but after you use the bathroom. Make sure you are wearing similar clothing each time you weigh yourself. Write down your weight every day on a chart so that you can keep track of it.

Throughout the day, ask yourself:
Is my energy level normal?
Do I get more short of breath when I am doing my everyday activities?
Are my clothes or shoes feeling tight?
Are my ankles or legs swelling?
Am I coughing more often? Does my cough sound wet?
Do I get short of breath at night or when I lie down?

If you are having new (or different) symptoms, ask yourself:

Did I eat something different than usual or try a new food?
Did I take all of my medicines the right way at the right times?

Diet and Fluids
Your health care provider may ask you to limit how much you drink.

When your heart failure is not very severe, you may not have to limit your fluids too much.
As your heart failure gets worse, you may be asked to limit fluids to 6 to 9 cups (1.5 to 2 liters) a day.

You will need to eat less salt. Salt can make you thirsty, and being thirsty can cause you to drink too much fluid. Extra salt also makes fluid stay in your body. Lots of foods that do not taste salty, or that you do not add salt to, still contain a lot of salt.

You may need to take a diuretic, or water pill.

Do not drink alcohol. Alcohol makes it harder for your heart muscles to work. Ask your provider what to do on special occasions where alcohol and foods you are trying to avoid will be served.

If you smoke, stop. Ask for help quitting if you need it. Do not let anybody smoke in your home.

Learn more about what you should eat to make your heart and blood vessels healthier.
Avoid fatty foods.
Stay away from fast-food restaurants.
Avoid some prepared and frozen foods.

Try to stay away from things that are stressful for you. If you feel stressed all the time, or if you are very sad, talk with your provider who can refer you to a counselor.

Taking Your Heart Drugs
Have your entire medicine prescriptions filled before you go home. It is very important that you take your medicines the way your provider told you to. Do not take any other medicines or herbs without asking your provider about them first.

Take your medicines with water. Do not take them with grapefruit juice, since it may change how your body absorbs certain medicines. Ask your provider or pharmacist if this will be a problem for you.

The medicines below are given to many people who have heart failure. Sometimes there is a reason they may not be safe to take, though. These medicines may help protect your heart. Talk with your provider if you are not already on any of these medicines:
Antiplatelet medicines (blood thinners) such as aspirin or clopidogrel (Plavix) to help keep your blood from clotting
Anticoagulant medicines (blood thinners) such as warfarin (Coumadin) to help keep your blood from clotting
Beta blocker and ACE inhibitor medicines as well as other medicines to lower your blood pressure and treat the heart muscle
Statins or other medicines to lower your cholesterol

Talk to your provider before changing the way you take your medicines. Never just stop taking these medicines for your heart, or any medicines you may be taking for diabetes, high blood pressure, or other medical conditions you have.

If you are taking a blood thinner, such as warfarin (Coumadin), you will need to have extra blood tests to make sure your dose is correct. Some other blood thinners do not require this.

Activity
Your provider may refer you to a cardiac rehabilitation program. There, you will learn how to slowly increase your exercise and how to take care of your heart disease. Make sure you avoid heavy lifting.

Make sure you know the warning signs of heart failure and of a heart attack. Know what to do when you have chest pain, or angina.

Always ask your provider before starting sexual activity again. Do not take sildenafil (Viagra), or vardenafil (Levitra), tadalafil (Cialis), or any herbal remedy for erection problems without checking first.

Make sure your home is set up to be safe and easy for you for you to move around in and avoid falls.

If you are unable to walk around very much, ask your provider for exercises you can do while you are sitting.

Follow-up
Make sure you get a flu shot and COVID-19 vaccine on a schedule recommended by your provider (usually every year). You may also need a pneumococcal vaccine (pneumonia shot). Ask your provider about this.

Your provider may call you to see how you are doing and to make sure you are checking your weight and taking your medicines.

You will need follow-up appointments at your provider's office.

You will likely need to have certain lab tests to check your sodium and potassium levels and monitor how your kidneys are working.

When to Call the Doctor
Contact your provider if:
You gain more than 2 pounds (lb) (1 kilogram, kg) in a day, or 5 lb (2 kg) in a week.
You are very tired and weak.
You are dizzy and lightheaded.
You are more short of breath when you are doing your normal activities.
You have new shortness of breath when you are sitting.
You need to sit up or use more pillows at night because you are short of breath when you are lying down.
You wake up 1 to 2 hours after falling asleep because you are short of breath.
You are wheezing and having trouble breathing.
You feel pain or pressure in your chest.
You have a cough that does not go away. It may be dry and hacking, or it may sound wet and bring up pink, foamy spit.
Your have swelling in your feet, ankles or legs.
You have to urinate a lot, particularly at night.
You have stomach pain or tenderness.
You have symptoms that you think may be from your medicines.
Your pulse, or heartbeat, gets very slow or very fast, or it is not steady.""",
        word_count=1050,
        reading_level="6th-8th grade",
    ),

    PatientDocument(
        doc_id="CARD_002",
        title="Taking warfarin",
        category="cardiology",
        topic="anticoagulation",
        source="MedlinePlus Medical Encyclopedia",
        source_url="https://medlineplus.gov/ency/patientinstructions/000292.htm",
        english_text="""Taking warfarin

Warfarin is a medicine that makes your blood less likely to form clots. It is important that you take warfarin exactly as you have been told. Changing how you take your warfarin, taking other medicines with it or at different times, and eating certain foods can change the way warfarin works in your body. If this happens, you may be more likely to form a clot or have bleeding problems.

What to Expect at Home
Warfarin is a medicine that makes your blood less likely to form clots. This may be important if:

You have already had blood clots in your leg, arm, heart, or brain.
Your health care provider is worried that a blood clot may form in your body. People who have a new heart valve, a large heart, a heart rhythm that is not normal, or other heart problems may need to take warfarin.
When you are taking warfarin, you may be more likely to bleed, even from activities you have always done.

Changing how you take your warfarin, taking other medicines, and eating certain foods all can change the way warfarin works in your body. If this happens, you may be more likely to form a clot or have bleeding problems.

Taking Warfarin
It is important that you take warfarin exactly as you have been told.

Take only the dose your provider has prescribed. If you miss a dose, contact your provider for advice.
If your pills look different from your last prescription, contact your provider or pharmacist right away. The tablets are different colors, depending on the dose. The dose is also marked on the pill.
Your provider will test your blood at regular visits. This is called an INR test or sometimes a PT test. The test helps make sure you are taking the right amount of warfarin to help your body.

Alcohol and some medicines can change how warfarin works in your body.

DO NOT drink alcohol while you are taking warfarin.
Talk with your provider before taking any other over-the-counter medicines, vitamins, supplements, cold medicines, antibiotics, or other medicines.
Tell all of your providers that you are taking warfarin. This includes doctors, nurses, and your dentist. Sometimes, you may need to stop or take less warfarin before having a procedure. Always talk to the provider who prescribed the warfarin before stopping or changing your dose.

Ask about wearing a medical alert bracelet or necklace that says you are taking warfarin. This will let providers who take care of you in an emergency know that you are taking this medicine.

Your Diet
Some foods can change the way warfarin works in your body. Make sure you check with your provider before making any big changes in your diet.

You do not have to avoid these foods, but try to eat or drink only small amounts of them. At the least, do not change the amount of these foods and products you eat day-to-day or week-to-week:

Mayonnaise and some oils, such as canola, olive, and soybean oils
Broccoli, Brussels sprouts, and raw green cabbage
Endive, lettuce, spinach, parsley, watercress, garlic, and scallions (green onions)
Kale, collard greens, mustard greens, and turnip greens
Cranberry juice and green tea
Fish oil supplements
Herbs used in herbal teas

Other Tips
Because being on warfarin can make you bleed more than usual:

You should avoid activities that could cause an injury or open wound, such as contact sports.
Use a soft toothbrush, waxed dental floss, and an electric razor. Be extra careful around sharp objects.
Prevent falls in your home by having good lighting and removing loose rugs and electric cords from pathways. Do not reach or climb for objects in the kitchen. Put things where you can get to them easily. Avoid walking on ice, wet floors, or other slippery or unfamiliar surfaces.

Make sure you look for unusual signs of bleeding or bruising on your body.

Look for bleeding from the gums, blood in your urine, bloody or dark stool, nosebleeds, or vomiting blood.
Women need to watch for extra bleeding during their period or between periods.
Dark red or black bruises may appear. If this happens, call your provider right away.

When to Call the Doctor
Contact your provider if you have:

A serious fall, or if you hit your head
Pain, discomfort, swelling at an injection or injury site
A lot of bruising on your skin
A lot of bleeding (such as nosebleeds or bleeding gums)
Bloody or dark brown urine or stool
Headache, dizziness, or weakness
A fever or other illness, including vomiting, diarrhea, or infection
You become pregnant or are planning to become pregnant""",
        word_count=720,
        reading_level="6th-8th grade",
    ),

    PatientDocument(
        doc_id="CARD_003",
        title="Heart attack – discharge",
        category="cardiology",
        topic="heart_attack_mi",
        source="MedlinePlus Medical Encyclopedia",
        source_url="https://medlineplus.gov/ency/patientinstructions/000090.htm",
        english_text="""Heart attack – discharge

A heart attack occurs when blood flow to a part of your heart is blocked long enough that part of the heart muscle is damaged or dies. This article discusses what you need to do to take care of yourself after you leave the hospital.

When You're in the Hospital
You were in the hospital because you had a heart attack. A heart attack occurs when blood flow to a part of your heart is blocked long enough that part of the heart muscle is damaged or dies.

What to Expect at Home
You may feel sad. You may feel anxious and as though you have to be very careful about what you do. All of these feelings are normal. They go away for most people after 2 or 3 weeks. You may also feel tired when you leave the hospital to go home.

Activity
You should know the signs and symptoms of angina.

You may feel pressure, squeezing, burning, or tightness in your chest. You may also notice these symptoms in your arms, shoulders, neck, jaw, throat, or back.
Some people also feel discomfort in their back, shoulders, and stomach area.
You may have indigestion or feel sick to your stomach.
You may feel tired and be short of breath, sweaty, lightheaded, or weak.
You may have angina during physical activity, such as climbing stairs or walking uphill, lifting, sexual activity, or when you are out in cold weather. It can also happen when you are resting or it can wake you up when you are sleeping.
Know how to treat your chest pain when it happens. Talk with your health care provider about what to do.

Take it easy for the first 4 to 6 weeks after your heart attack.

Avoid heavy lifting. Get some help with household chores if you can.
Take 30 to 60 minutes to rest in the afternoon for first 4 to 6 weeks. Try to go to bed early and get plenty of sleep.
Before starting to exercise, your provider may have you do an exercise test and recommend an exercise plan. This may happen before you leave the hospital or soon afterward. Do not change your exercise plan before talking with your provider.
Your provider may refer you to cardiac rehabilitation program. There, you will learn how to slowly increase your exercise and how to take care of your heart disease.
If you had angiography, follow your provider's instructions to care for the puncture site.
You should be able to talk comfortably when you are doing any activity, such as walking, setting the table, and doing laundry. If you cannot, stop the activity.

Ask your provider about when you can return to work. Expect to be away from work for at least a week.

Talk to your provider before engaging in sexual activity. Ask your provider when it is OK to start again. Do not take Viagra, Levitra, Cialis or any herbal remedy for erection problems without checking with your provider first.

How long you will have to wait to return to your normal activities will depend on:

Your physical condition before your heart attack
The size of your heart attack
If you had complications
The overall speed of your recovery

Diet and Lifestyle
Do not drink any alcohol for at least 2 weeks. Ask your provider when you may start and for other guidance about how much alcohol is safe for you.

If you smoke, stop. Ask your provider for help quitting if you need it. Do not let anybody smoke in your home, since second-hand smoke can harm you. Try to stay away from things that are stressful for you. If you are feeling stressed all the time, or if you are feeling very sad, talk with your provider. They can refer you to a counselor.

Learn more about what you should eat to make your heart and blood vessels healthier.

Avoid salty foods.
Stay away from fast food restaurants.

Taking Your Heart Medicines
Have your prescriptions filled before you go home. It is very important that you take your medicines the way your provider told you to. Do not take any other medicines or herbal supplements without asking your provider first if they are safe for you.

Take your medicines with water. Do not take them with grapefruit juice, since it may change how your body absorbs certain medicines. Ask your provider or pharmacist for more information about this.

The medicines below are given to most people after they have had a heart attack. Sometimes there is a reason they may not be safe to take, though. These medicines help prevent another heart attack. Talk with your provider if you are not already on any of these medicines:

Antiplatelets medicines (blood thinners), such as aspirin, clopidogrel (Plavix), warfarin (Coumadin), prasugrel (Efient), or ticagrelor (Brilinta) to help keep your blood from clotting
Beta-blockers and ACE inhibitor medicines to help protect your heart
Statins or other drugs to lower your cholesterol
Calcium-channel blockers
Nitroglycerin

Do not suddenly stop taking these medicines for your heart. Do not stop taking medicines for your diabetes, high blood pressure, or any other medical conditions you may have without talking with your provider first.

If you are taking a blood thinner such as warfarin (Coumadin), you may need to have extra blood tests on a regular basis to make sure your dose is correct.

When to Call the Doctor
Contact your provider if you feel:

Pain, pressure, tightness, or heaviness in your chest, arm, neck, or jaw
Shortness of breath
Gas pains or indigestion
Numbness in your arms
Sweaty, or if you lose color
Lightheaded

Changes in your angina may mean your heart disease is getting worse. Contact your provider if your angina:

Becomes stronger
Happens more often
Lasts longer
Occurs when you are not active or when you are resting
Medicines do not help ease your symptoms as well as they used to""",
        word_count=870,
        reading_level="6th-8th grade",
    ),

    PatientDocument(
        doc_id="CARD_004",
        title="High blood pressure - what to ask your doctor",
        category="cardiology",
        topic="hypertension",
        source="MedlinePlus Medical Encyclopedia",
        source_url="https://medlineplus.gov/ency/patientinstructions/000226.htm",
        english_text="""High blood pressure - what to ask your doctor

When your heart pumps blood into your arteries, the pressure of the blood against the artery walls is called your blood pressure. Your blood pressure is given as two numbers: systolic over diastolic blood pressure. Your systolic blood pressure is the highest blood pressure during the course of your heart beat cycle. Your diastolic blood pressure is the lowest pressure.

When your blood pressure gets too high, it puts extra stress on your heart and blood vessels. If your blood pressure stays high all the time, you will be at a higher risk for heart attacks and other vascular (blood vessel) diseases, strokes, kidney disease, and other health problems.

Below are questions you may want to ask your health care provider to help you take care of your blood pressure.

Questions
How can I change the way I live to lower my blood pressure?

What is a heart-healthy diet? Is it OK to ever eat something that is not heart healthy? What are some ways to eat healthy when I go to a restaurant?
Do I need to limit how much salt I use? Are there other spices that I can use to make my food taste good?
Is it OK to drink alcohol? How much is OK?
What can I do to stop smoking? Is it OK to be around other people who are smoking?
Should I check my blood pressure at home?

What type of equipment should I buy? Where can I learn how to use it?
How often do I need to check my blood pressure? Should I write it down and bring it to my next visit?
If I cannot check my own blood pressure, where else can I have it checked?
What should my blood pressure reading be? Should I rest before taking my blood pressure?
When should I contact my provider?
What is my cholesterol? Do I need to take medicines for it?

Is it OK to be sexually active? Is it safe to use sildenafil (Viagra), or tadalafil (Cialis), vardenafil (Staxyn), or avanafil (Stendra) for erection problems?

What medicines am I taking to treat high blood pressure?

Do they have any side effects? What should I do if I miss a dose?
Is it ever safe to stop taking any of these medicines on my own?
How much activity can I do?

Do I need to have a stress test before I exercise?
Is it safe for me to exercise on my own?
Should I exercise inside or outside?
Which activities should I start with? Are there activities or exercises that are not safe for me?
How long and how hard can I exercise?
What are the warning signs that I should stop exercising?""",
        word_count=420,
        reading_level="6th-8th grade",
    ),

    # =========================================================================
    # DIABETES - Verified from MedlinePlus
    # =========================================================================
    PatientDocument(
        doc_id="DIAB_001",
        title="Type 2 diabetes - what to ask your doctor",
        category="diabetes",
        topic="type2_diabetes_basics",
        source="MedlinePlus Medical Encyclopedia",
        source_url="https://medlineplus.gov/ency/patientinstructions/000217.htm",
        english_text="""Type 2 diabetes - what to ask your doctor

Type 2 diabetes, once diagnosed, is a lifelong disease that causes a high level of sugar (glucose) in your blood. It can damage your organs. It can also lead to a heart attack or stroke and many other health problems. You can take steps to control your symptoms, prevent damage due to diabetes, and make your life better.

Below are questions you may want to ask your health care provider to help you take care of your diabetes.

Questions
Ask your provider to check the nerves, skin, and pulses in your feet. Also ask these questions:

How often should I check my feet? What should I do when I check them? What problems should I call my provider about?
Who should trim my toenails? Is it OK if I trim them?
How should I take care of my feet every day? What type of shoes and socks should I wear?
Should I see a foot doctor (podiatrist)?
Ask your provider about getting exercise, including:

Before I start, do I need to have my heart checked? My eyes? My feet?
What type of exercise program should I do? What type of activities should I avoid?
When should I check my blood sugar when I exercise? What should I bring with me when I exercise? Should I eat before or during exercise? Do I need to adjust my medicines when I exercise?
When should I next have an eye doctor (optometrist or ophthalmologist) check my eyes? What eye problems should I contact my eye doctor about?

Ask your provider about meeting with a dietitian. Questions for the dietitian may include:

What foods increase my blood sugar the most?
What foods can help me with my weight loss goals?
Ask your provider about your diabetes medicines:

When should I take them?
What should I do if I miss a dose?
Are there any side effects?
How often should I check my blood sugar level at home? Should I do it at different times of the day? What is too low? What is too high? What should I do if my blood sugar is too low or too high?

Should I get a medical alert bracelet or necklace? Should I have glucagon at home?

Ask your provider about symptoms that you are having if they have not been discussed. Tell your provider about blurred vision, skin changes, depression, reactions at injection sites, sexual dysfunction, tooth pain, muscle pain, or nausea.

Ask your provider about other tests you may need, such as cholesterol, HbA1C, and a urine and blood test to check for kidney problems.

Ask your provider about vaccinations you should have like the COVID-19 vaccine, flu shot, hepatitis B, or pneumococcal (pneumonia) vaccines.

How should I take care of my diabetes when I travel?

Ask your provider how you should take care of your diabetes when you are sick:

What should I eat or drink?
How should I take my diabetes medicines?
How often should I check my blood sugar?
When should I contact the provider?""",
        word_count=480,
        reading_level="6th-8th grade",
    ),

    PatientDocument(
        doc_id="DIAB_002",
        title="Low blood sugar - self-care",
        category="diabetes",
        topic="hypoglycemia",
        source="MedlinePlus Medical Encyclopedia",
        source_url="https://medlineplus.gov/ency/patientinstructions/000085.htm",
        english_text="""Low blood sugar - self-care

Low blood sugar is a condition that occurs when your blood sugar (glucose) is lower than normal. Low blood sugar may occur in people with diabetes who are taking insulin or certain other medicines to control their diabetes. Low blood sugar can cause dangerous symptoms. Learn how to recognize the symptoms of low blood sugar and how to prevent them.

What is Low Blood Sugar?
Low blood sugar is called hypoglycemia. A blood sugar level below 70 mg/dL (3.9 mmol/L) is low and can harm you. A blood sugar level below 54 mg/dL (3.0 mmol/L) is a cause for immediate action.

You are at risk for low blood sugar if you have diabetes and are taking any of the following diabetes medicines:

Insulin
Glyburide (Micronase), glipizide (Glucotrol), glimepiride (Amaryl), repaglinide (Prandin), or nateglinide (Starlix)
You are also at increased risk of having low blood sugar if you have had previous low blood sugar levels.

Recognizing Low Blood Sugar
Know how to tell when your blood sugar is getting low. Symptoms include:

Weakness or feeling tired
Shaking
Sweating
Headache
Hunger
Feeling uneasy, nervous, or anxious
Feeling cranky
Trouble thinking clearly
Double or blurry vision
Fast or pounding heartbeat
Sometimes your blood sugar may be too low even if you do not have symptoms. If it gets too low, you may:

Faint
Have a seizure
Go into a coma
Some people who have had diabetes for a long time stop being able to sense low blood sugar. This is called hypoglycemic unawareness. Ask your health care provider if wearing a continuous glucose monitor and sensor can help you detect when your blood sugar is getting too low in order to help prevent symptoms.

Check Your Blood Sugar Often
Talk with your provider about when you should check your blood sugar every day. People who have low blood sugar need to check their blood sugar more often.

The most common causes of low blood sugar are:

Taking your insulin or diabetes medicine at the wrong time
Taking too much insulin or diabetes medicine
Taking insulin to correct high blood sugar without eating any food
Not eating enough during meals or snacks after you have taken insulin or diabetes medicine
Skipping meals (this may mean that your dose of long-acting insulin is too high, so you should talk to your provider)
Waiting too long after taking your medicine to eat your meals
Exercising a lot or at a time that is unusual for you
Not checking your blood sugar or not adjusting your insulin dose before exercising
Drinking alcohol

Preventing Low Blood Sugar
Preventing low blood sugar is better than having to treat it. Always have a source of fast-acting sugar with you.

When you exercise, check your blood sugar levels. Make sure you have snacks with you.
Talk to your provider about reducing insulin doses on days that you exercise.
Ask your provider if you need a bedtime snack to prevent low blood sugar overnight. Protein snacks may be best.
Do not drink alcohol without eating food. Women should limit alcohol to 1 drink a day and men should limit alcohol to 2 drinks a day. Family and friends should know how to help. They should know:

The symptoms of low blood sugar and how to tell if you have them.
How much and what kind of food they should give you.
When to call for emergency help.
How to inject glucagon, a hormone that increases your blood sugar. Your provider will tell you when to use this medicine.
If you have diabetes, always wear a medical alert bracelet or necklace. This helps emergency medical workers know you have diabetes.

When Your Blood Sugar Gets Low
Check your blood sugar whenever you have symptoms of low blood sugar. If your blood sugar is below 70 mg/dL, treat yourself right away.

1. Eat something that has about 15 grams (g) of carbohydrates. Examples are:

4 glucose tablets
One half cup (4 ounces or 120 mL) of fruit juice or regular, non-diet soda
5 or 6 hard candies
1 tablespoon (tbsp) or 15 mL of sugar, plain or dissolved in water
1 tbsp (15 mL) of honey or syrup

2. Wait about 15 minutes before eating any more. Be careful not to eat too much. This can cause high blood sugar and weight gain.

3. Check your blood sugar again.

4. If you do not feel better in 15 minutes and your blood sugar is still lower than 70 mg/dL (3.9 mmol/L), eat another snack with 15 g of carbohydrates.

You may need to eat a snack with carbohydrates and protein if your blood sugar is in a safer range -- over 70 mg/dL (3.9 mmol/L) -- and your next meal is more than an hour away.

Ask your provider how to manage this situation. If these steps for raising your blood sugar do not work, contact your provider right away.

Talk to Your Doctor or Nurse
If you use insulin and your blood sugar is frequently or consistently low, ask your provider or nurse if you:

Are injecting your insulin the right way
Need a different type of needle
Should change how much insulin you take
Should change the kind of insulin you take
Do not make any changes without talking to your provider or nurse first.

Sometimes hypoglycemia can be due to accidently taking the wrong medicines. Check your medicines with your pharmacist.

When to Call the Doctor
If signs of low blood sugar do not improve after you have eaten a snack that contains sugar, have someone drive you to the emergency room or call 911 or the local emergency number. Do not drive when your blood sugar is low.

Get medical help right away for a person with low blood sugar if the person is not alert or cannot be woken up as this is a medical emergency.""",
        word_count=920,
        reading_level="6th-8th grade",
    ),

    # =========================================================================
    # RESPIRATORY - Verified from MedlinePlus
    # =========================================================================
    PatientDocument(
        doc_id="RESP_001",
        title="Asthma - quick-relief drugs",
        category="respiratory",
        topic="asthma_management",
        source="MedlinePlus Medical Encyclopedia",
        source_url="https://medlineplus.gov/ency/patientinstructions/000008.htm",
        english_text="""Asthma - quick-relief drugs

Asthma quick-relief medicines work fast to control asthma symptoms. You take them when you are coughing, wheezing, having trouble breathing, or having an asthma attack. They are also called rescue medicines.

Many of these medicines are called "bronchodilators" because they open (dilate) and help relax the muscles of your airways (bronchi).

You and your health care provider can make a plan for the quick-relief medicines that work for you. This plan will include when you should take them and how much you should take.

Plan ahead. Make sure you do not run out. Bring enough medicine with you when you travel.

Short-acting Beta-agonists
Short-acting beta-agonists are the most common quick-relief medicines for treating asthma attacks and are considered to be bronchodilators.

They can be used just before exercising to help prevent asthma symptoms caused by exercise. They work by relaxing the muscles of your airways, and this lets you breathe better during an attack.

Tell your provider if you are using quick-relief medicines twice a week or more to control your asthma symptoms. Your asthma may not be under control, and your provider may need to change your dose of daily control medicines.

Some quick-relief asthma medicines include:

Albuterol (ProAir HFA, Proventil HFA, Ventolin HFA)
Levalbuterol (Xopenex HFA)
Metaproterenol
Terbutaline
Short-acting beta-agonists may cause these side effects:

Anxiety.
Fast and irregular heartbeats. Call your provider right away if you have this side effect.
Headache.
Restlessness.
Tremor (your hand or another part of your body may shake).
Oral Steroids
Your provider might prescribe oral steroids when you have an asthma attack that is not going away. These are medicines that you take by mouth as pills, capsules, or liquids.

Oral steroids are not quick-relief medicines but are often given for 7 to 14 days when your symptoms flare-up.

Oral steroids include:

Methylprednisolone
Prednisone
Prednisolone""",
        word_count=310,
        reading_level="6th-8th grade",
    ),

    PatientDocument(
        doc_id="RESP_002",
        title="COPD - what to ask your doctor",
        category="respiratory",
        topic="copd_care",
        source="MedlinePlus Medical Encyclopedia",
        source_url="https://medlineplus.gov/ency/patientinstructions/000215.htm",
        english_text="""COPD - what to ask your doctor

Chronic obstructive pulmonary disease (COPD) damages your lungs. This can make it hard for you to get enough oxygen and clear carbon dioxide from your lungs. While there is no cure for COPD, you can do many things to regulate your symptoms and make your life better.

Below are some questions you may want to ask your health care provider to help you take care of your lungs.

Questions
What will make my COPD worse?

How can I avoid things that can make my COPD worse?
How can I prevent getting a lung infection?
How can I get help quitting smoking?
Will fumes, dust, or having pets make my COPD worse?
What are some signs that my breathing is getting worse and I should contact my provider? What should I do when I feel I am not breathing well enough?

Am I taking my COPD medicines the right way?

What medicines should I be taking every day (called controller medicines)? What should I do if I miss a day or a dose?
Which medicines should I take when I am short of breath (called quick-relief or rescue medicines)? Is it OK to use these medicines every day?
What are the side effects of my medicines? For what side effects should I contact my provider?
Am I using my inhaler the right way? Should I be using a spacer? How will I know when my inhalers are getting empty?
When should I use my nebulizer and when should I use my inhaler?
What sort of changes should I make around my home?

What sort of changes do I need to make at work?

What shots or vaccinations do I need?

Do I need oxygen? If yes, do I need it all times?

Are there changes in my diet that will help my COPD?

What do I need to do when I am planning to travel?

Will I need oxygen on the airplane? How about at the airport?
What medicines should I bring?
Who should I contact if my COPD gets worse?
What are some exercises I can do to keep my muscles strong, even if I cannot walk around very much?

Should I consider pulmonary rehabilitation?

How can I save some of my energy around the house?

Am I at higher risk for COVID-19 or other illness? How should I protect myself?""",
        word_count=370,
        reading_level="6th-8th grade",
    ),

    # =========================================================================
    # MEDICATIONS - Verified from MedlinePlus
    # =========================================================================
    PatientDocument(
        doc_id="MED_001",
        title="ACE inhibitors",
        category="medications",
        topic="blood_pressure_medication",
        source="MedlinePlus Medical Encyclopedia",
        source_url="https://medlineplus.gov/ency/patientinstructions/000087.htm",
        english_text="""ACE inhibitors

Angiotensin-converting enzyme (ACE) inhibitors are medicines. They treat heart, blood vessel, and kidney problems.

How ACE inhibitors help
ACE inhibitors are used to treat heart disease and high blood pressure. These medicines make your heart work less hard by lowering your blood pressure. This keeps some kinds of heart disease from getting worse. Most people who have heart failure take one of these medicines or similar medicines.

These medicines treat high blood pressure, strokes, or heart attacks. They may help lower your risk for stroke or heart attack.

They are also used to treat kidney problems. This can help keep your kidneys from getting worse or even improve the function of your kidneys, especially if you have diabetes. If you have these problems, ask your health care provider if you should be taking these medicines.

Types of ACE inhibitors
There are many different names and brands of ACE inhibitors. Most work as well as another. Side effects may be different for different ones.

Taking Your ACE inhibitors
ACE inhibitors are pills that you take by mouth. Take all of your medicines as your provider told you to. Follow up with your provider regularly. Your provider will check your blood pressure and do blood tests to make sure the medicines are working properly. Your provider may change your dose from time to time. In addition:

Try to take your medicines at the same time each day.
Don't stop taking your medicines without talking to your provider first.
Plan ahead so that you do not run out of medicine. Make sure you have enough with you when you travel.
Before taking ibuprofen (Advil, Motrin) or aspirin, talk to your provider.
Tell your provider what other medicines you are taking, including anything you bought without a prescription, diuretics (water pills), potassium pills, or herbal or dietary supplements.
Don't take ACE inhibitors if you are planning to become pregnant, are pregnant, or are breastfeeding. Contact your provider promptly if you become pregnant when you are taking these medicines.
Side effects
Side effects from ACE inhibitors are unusual.

You may have a dry cough. This may go away after a while. It also may start after you have been taking the medicine for some time. Tell your provider if you develop a cough. Sometimes reducing your dose helps. But sometimes, your provider will switch you to a different medicine. Do not lower your dose without talking with your provider first.

You may feel dizzy or lightheaded when you start taking these medicines, or if your provider increases your dose. Standing up slowly from a chair or your bed may help. If you have a fainting spell, contact your provider right away.

Other side effects include:

Headache
Fatigue
Loss of appetite
Upset stomach
Diarrhea
Numbness
Fever
Skin rashes or blisters
Joint pain
If your tongue or lips swell, contact your provider right away, or go to the emergency room. You may be having a serious allergic reaction to the medicine. This is very rare.

When to Call the Doctor
Contact your provider if you are having any of the side effects listed above. Also contact your provider if you are having any other unusual symptoms.""",
        word_count=480,
        reading_level="6th-8th grade",
    ),

    # =========================================================================
    # EMERGENCY CARE - Verified from MedlinePlus
    # =========================================================================
    PatientDocument(
        doc_id="EMER_001",
        title="Stroke - discharge",
        category="emergency_care",
        topic="stroke_discharge",
        source="MedlinePlus Medical Encyclopedia",
        source_url="https://medlineplus.gov/ency/patientinstructions/000132.htm",
        english_text="""Stroke - discharge

You were in the hospital after having a stroke. A stroke happens when blood flow to part of the brain stops.

At home follow your health care provider's instructions on self-care. Use the information below as a reminder.

When You're in the Hospital
First, you received treatment to prevent further damage to the brain, and to help the heart, lungs, and other important organs heal.

After you were stable, your providers did testing and started treatment to help you recover from the stroke and prevent a future stroke. You may have stayed in a special unit that helps people recover after a stroke.

What to Expect at Home
Because of possible injury to the brain from the stroke, you may notice problems with:

Changes in behavior
Doing easy tasks
Memory
Moving one side of the body
Muscle spasms
Paying attention
Sensation or awareness of one part of the body
Swallowing
Talking or understanding others
Thinking
Seeing to one side (hemianopsia)
You may need help with daily activities you used to do alone before the stroke. You may have physical or occupational therapy to restore your ability to perform these functions.

Depression after a stroke is fairly common as you learn to live with the changes. It may develop soon after the stroke or up to 2 years after the stroke.

Do not drive your car without your provider's permission.

Moving Around
Moving around and doing normal tasks may be hard after a stroke.

Make sure your home is safe. Ask your provider, therapist, or nurse about making changes in your home to make it easier to do everyday activities.

Find out about what you can do to prevent falls and keep your bathroom safe to use.

Family and caregivers may need to help with:

Exercises to keep your elbows, shoulders, and other joints loose
Watching for joint tightening (contractures)
Making sure splints are used in the correct way
Making sure arms and legs are in a good position when sitting or lying
If you or your loved one is using a wheelchair, follow-up visits to make sure it fits well are important to prevent skin ulcers.

Check every day for pressure sores at the heels, ankles, knees, hips, tailbone, and elbows.
Change positions in the wheelchair several times per hour during the day to prevent pressure ulcers.
If you have problems with spasticity, learn what makes it worse. You or your caregiver can learn exercises to keep your muscles loose.
Learn how to prevent pressure ulcers.
Thinking and Speaking
Tips for making clothing easier to put on and take off are:

Velcro is much easier than buttons and zippers. All buttons and zippers should be on the front of a piece of clothing.
Use pullover clothes and slip-on shoes.
People who have had a stroke may have speech or language problems. You may have been referred to a speech therapist to help with these problems. Tips for family and caregivers to improve communication include:

Keep distractions and noise down. Keep your voice lower. Move to a quieter room. Do not shout.
Allow plenty of time for the person to answer questions and understand instructions. After a stroke, it takes longer to process what has been said.
Use simple words and sentences, speak slowly. Ask questions in a way that can be answered with a yes or no. When possible, give clear choices. Do not give too many options.
Break down instructions into small and simple steps.
Repeat if needed. Use familiar names and places. Announce when you are going to change the subject.
Make eye contact before touching or speaking if possible.
Use props or visual prompts when possible. Do not give too many options. You may be able to use pointing or hand gestures or drawings. Use an electronic device, such as a tablet computer or cell phone, to show pictures to help with communication.
Bowel Care
Nerves that help the bowels work smoothly can be damaged after a stroke. Have a routine. Once you find a bowel routine that works, stick to it:

Pick a regular time, such as after a meal or a warm bath, to try to have a bowel movement.
Be patient. It may take 15 to 45 minutes to have bowel movements.
Try gently rubbing your stomach to help stool move through your colon.
Avoid constipation:

Drink more fluids.
Stay active or become more active as much as possible.
Eat foods with lots of fiber.
Ask your provider about medicines you are taking that may cause constipation (such as medicines for depression, pain, bladder control, and muscle spasms).

Tips for Taking Medicines
Have all of your prescriptions filled before you go home. It is very important that you take your medicines the way your provider told you to. Do not take any other medicines, supplements, vitamins, or herbs without asking your provider about them first.

You may be given one or more of the following medicines. These are meant to regulate your blood pressure or cholesterol, and to keep your blood from clotting. They may help prevent another stroke:

Antiplatelet medicines (aspirin or clopidogrel) help keep your blood from clotting.
Blood thinners such as warfarin or novel oral anticoagulants (NOACs, such as apixaban, rivaroxaban, and others)
Calcium channel blockers, diuretics (water pills), and ACE inhibitor medicines regulate your blood pressure and protect your heart.
Statins lower your cholesterol.
If you have diabetes, regulate your blood sugar at the level your provider recommends.
Do not stop taking any of these medicines.

Some blood thinners, such as warfarin (Coumadin), will require you to have extra blood tests done.

Staying Healthy
If you have problems with swallowing, you must learn to follow a special diet that makes eating safer. The signs of swallowing problems are choking or coughing when eating. Learn tips to make feeding and swallowing easier and safer.

Avoid salty and fatty foods and stay away from fast food restaurants to make your heart and blood vessels healthier.

Limit how much alcohol you drink to a maximum of 1 drink a day if you are a woman and 2 drinks a day if you are a man. Ask your provider if it is OK for you to drink alcohol.

Keep up to date with your vaccinations. Get a flu shot every year. Ask your provider if you need a vaccination to prevent pneumococcal infections (sometimes called a pneumonia shot) and a COVID-19 vaccine.

Do not smoke. Ask your provider for help quitting if you need to. Do not let anybody smoke in your home.

Try to stay away from stressful situations. If you feel stressed all the time or feel very sad, talk with your provider.

If you feel sad or depressed at times, talk to family or friends about this. Ask your provider about seeking professional help.

When to Call the Doctor
Contact your provider if you have:

Problems taking medicines for muscle spasms
Problems moving your joints (joint contracture)
Problems moving around or getting out of your bed or chair
Skin sores or redness
Pain that is becoming worse
Recent falls
Choking or coughing when eating
Signs of a bladder infection (fever, burning when you urinate, or frequent urination)
Call 911 or the local emergency number if the following symptoms develop suddenly or are new:

Numbness or weakness of the face, arm, or leg
Blurry or decreased vision
Not able to speak or understand
Dizziness, loss of balance, or falling
Severe headache""",
        word_count=1120,
        reading_level="6th-8th grade",
    ),

    # =========================================================================
    # SURGICAL - Verified from MedlinePlus
    # =========================================================================
    PatientDocument(
        doc_id="SURG_001",
        title="Surgical wound care - open",
        category="surgical",
        topic="wound_care",
        source="MedlinePlus Medical Encyclopedia",
        source_url="https://medlineplus.gov/ency/patientinstructions/000040.htm",
        english_text="""Surgical wound care - open

An incision is a cut through the skin that is made during surgery. It is also called a surgical wound. Some incisions are small, others are long. The size of the incision depends on the kind of surgery you had.

Sometimes, an incision breaks open. This may happen along the entire cut or just part of it. Your surgeon may decide not to close it again with sutures (stitches).

What to Expect at Home
If your surgeon does not close your wound again with sutures, you need to care for it at home, since it may take time to heal. The wound will heal from the bottom to the top. A dressing helps absorb drainage and keep the skin from closing before the wound underneath fills in.

Proper Handwashing
It is important to clean your hands before you change your dressing. You can use an alcohol-based cleanser. Or, you can wash your hands using these steps:

Take all jewelry off your hands.
Wet your hands, pointing them downward under warm running water.
Add soap and wash your hands for 15 to 30 seconds (sing "Happy Birthday" or the "Alphabet Song" one time through). Clean under your nails also.
Rinse well.
Dry with a clean towel.
Removing the Old Dressing
Your health care provider will tell you how often to change your dressing. To prepare for the dressing change:

Clean your hands before touching the dressing.
Make sure you have all the supplies handy.
Have a clean work surface.
Remove the old dressing:

Carefully loosen the tape from your skin.
Use a clean (not sterile) medical glove to grab the old dressing and pull it off.
If the dressing sticks to the wound, wet it and try again, unless your provider instructed you to pull it off dry.
Put the old dressing in a plastic bag and set it aside.
Clean your hands again after you take off the old dressing.
Caring for the Wound
You may use a gauze pad or soft cloth to clean the skin around your wound:

Use a normal saline solution (salt water) or mild soapy water.
Soak the gauze or cloth in the saline solution or soapy water, and gently dab or wipe the skin with it.
Try to remove all drainage and any dried blood or other matter that may have built up on the skin.
Do not use skin cleansers, alcohol, peroxide, iodine, or soap with antibacterial chemicals. These can damage the wound tissue and slow healing.
Your provider may also ask you to irrigate, or wash out, your wound:

Fill a syringe with salt water or soapy water, whichever your provider recommends.
Hold the syringe 1 to 6 inches (2.5 to 15 centimeters) away from the wound. Spray hard enough into the wound to wash away drainage and discharge.
Use a clean soft, dry cloth or piece of gauze to carefully pat the wound dry.
Do not put any lotion, cream, or herbal remedies on or around your wound, unless your provider has said it is OK.

Putting on the New Dressing
Place the clean dressing on the wound as your provider taught you to. You may be using a wet-to-dry dressing that requires moistening the dressing before applying it.

Clean your hands when you are finished.

Throw away the old dressing and other used supplies in a waterproof plastic bag. Close it tightly then double it before putting it in the trash.

Wash any soiled laundry from the dressing change separately from other laundry. Ask your provider if you need to add bleach to the wash water.

Use a dressing only once. Never reuse it.

When to Call the Doctor
Contact your provider if:

There is more redness, pain, swelling, or bleeding at the wound site.
The wound is larger or deeper, or it looks dried out or dark.
The drainage coming from or around the wound increases or becomes thick, tan, green, or yellow, or smells bad (which indicates pus).
Your temperature is 100.5°F (38°C) or higher.""",
        word_count=610,
        reading_level="6th-8th grade",
    ),

    PatientDocument(
        doc_id="SURG_002",
        title="Hip replacement - discharge",
        category="surgical",
        topic="hip_replacement",
        source="MedlinePlus Medical Encyclopedia",
        source_url="https://medlineplus.gov/ency/patientinstructions/000169.htm",
        english_text="""Hip replacement - discharge

You had surgery to replace all or part of your hip joint with an artificial joint called a prosthesis. This article tells you what you need to do to care for your new hip when you leave the hospital.

When You're in the Hospital
You had a hip joint replacement surgery to replace all or part of your hip joint with an artificial joint. This artificial joint is called a prosthesis.

What to Expect at Home
By the time you go home, you should be able to walk with a walker or crutches without needing much help. Most people do not need them after 2 to 4 weeks. Check with your surgeon about when to stop using crutches.

You should also be able to dress yourself with only a little help and be able to get into and out of your bed or a chair by yourself. You should also be able to use the toilet without much help.

You will need to be careful that you do not dislocate your artificial hip, especially in the first few months after surgery. You will need to learn exercises that make your new hip stronger and take special precautions.

You will need to have someone with you at home for 1 to 2 weeks after you leave the hospital or rehab center. You will need help preparing meals, bathing, moving around the house, and doing other daily activities.

Over time, you should be able to return to your former level of activity. You will need to avoid some sports, such as downhill skiing or contact sports like football and soccer. But you should be able to do low impact activities, such as hiking, gardening, swimming, playing tennis, and golfing.

Home Setup
Your bed should be low enough for your feet to touch the floor when you sit on the edge of the bed. Your bed should also be high enough so that your hips are higher than your knees when you sit on the edge. You may not need a hospital bed, but your mattress should be firm.

Keep tripping hazards out of your home.

Learn to prevent falls. Remove loose wires or cords from areas you walk through to get from one room to another. Remove loose throw rugs. Do not keep small pets in your home. Fix any uneven flooring in doorways. Use good lighting.
Make your bathroom safer. Put hand rails in the bathtub or shower and next to the toilet. Place a slip-proof mat in the bathtub or shower.
Do not carry anything when you are walking around. You may need your hands to help you balance. You can attach a pocket or holder to your walker or crutches to hold things instead.
Put things where they are easy to reach.

Place a chair with a firm back in the kitchen, bedroom, bathroom, and other rooms you will use. This way, you can sit when you do your daily tasks.

Set up your home so that you do not have to climb steps. Some tips are:

Set up a bed or use a bedroom on the first floor.
Have a bathroom or a portable commode on the same floor where you spend most of your day.
Activity
You will need to be careful to not dislocate your new hip when you are walking, sitting, lying down, dressing, taking a bath or shower, and doing other activities. Avoid sitting in a low chair or soft sofa.

Keep moving and walking once you get home. Try not to sit for more than 45 minutes at a time. Get up and move around after 45 minutes if you will be sitting longer.

Do not put your full weight on your side with the new hip until your surgeon tells you it is OK. Start out with short periods of activity, and then gradually increase them. Your surgeon or physical therapist will give you exercises to do at home.

Use your crutches or walker for as long as you need them. Check with your surgeon before you stop using them.

After a few days you may be able to do simple household chores. Do not try to do heavier chores, such as vacuuming or laundry. Remember, you will get tired quickly at first. Do not squat down to clean.

Wear a small fanny pack or backpack, or attach a basket or strong bag to your walker, so that you can keep small household items, like a phone and notepad, with you.

Wound Care
Keep your dressing (bandage) on your wound clean and dry. You may change the dressing according to when your surgeon told you to change it. Be sure to change it if it gets dirty or wet. Follow these steps when you change your dressing:

Wash your hands well with soap and water.
Remove the dressing carefully. Do not pull hard. If you need to, soak some of the dressing with sterile water or saline to help loosen it.
Soak some clean gauze with saline and wipe from one end of the incision to the other. Do not wipe back and forth over the same area.
Dry the incision the same way with clean, dry gauze. Wipe or pat in just one direction.
Check your wound for signs of infection. These include severe swelling and redness and drainage that has a bad odor.
Apply a new dressing the way you were shown.
Sutures (stitches) or staples will be removed about 10 to 14 days after surgery. Do not shower until 3 to 4 days after your surgery, or when your surgeon told you to shower. When you can shower, let water run over your incision but do not scrub it or let the water beat down on it. Do not soak in a bathtub, hot tub, or swimming pool.

You may have bruising around your wound. This is normal, and it will go away on its own. The skin around your incision may be a little red. This is normal too.

Self-care
Your surgeon will give you a prescription for pain medicines. Get it filled when you go home so you have it when you need it. Take your pain medicine when you start having pain. Waiting too long to take it will allow your pain to get more severe than it should.

In the early part of your recovery, taking pain medicine about 30 minutes before you increase your activity or do your physical therapy can help control pain.

You may be asked to wear special compression stockings on your legs for about 6 weeks. These will help prevent blood clots from forming. You may also need to take blood thinners for 2 to 4 weeks to lower your risk for blood clots. Take all your medicines the way your surgeon told you to. It can make your bruise go away more easily.

Your surgeon will tell you when it is OK to start sexual activity again.

People who have a prosthesis, such as an artificial joint, need to carefully protect themselves against infection. It used to be recommended that you need to take antibiotics before any dental work or invasive medical procedures, however, the recommendation has been changed except for high risk patients. Make sure to check with your surgeon, and tell your dentist or other surgeons about your hip replacement.

When to Call the Doctor
Contact your surgeon if you have:

A sudden increase in pain
Chest pain or shortness of breath
Frequent urination or burning when you urinate
Redness or increasing pain around your incision
Drainage from your incision
Blood in your stools, or your stools turn dark
Swelling in one of your legs (it will be red and warmer than the other leg)
Pain in your calf
Fever greater than 101°F (38.3°C)
Pain that is not controlled by your pain medicines
Nosebleeds or blood in your urine or stools if you are taking blood thinners
Also contact your surgeon if you:

Cannot move your hip as much as you could before
Fall or hurt your leg on the side that had surgery
Have increased pain in your hip
Have difficulty with walking and bending your hip
Feel like your hip have slipped or out of position""",
        word_count=1280,
        reading_level="6th-8th grade",
    ),
]


# =============================================================================
# DOCUMENT MANAGEMENT FUNCTIONS
# =============================================================================

def save_documents_to_json(documents: list[PatientDocument], filename: str = "source_documents.json"):
    """Save documents to JSON file."""
    filepath = DOCUMENTS_DIR / filename
    data = [doc.to_dict() for doc in documents]
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(documents)} documents to {filepath}")
    return filepath


def load_documents_from_json(filename: str = "source_documents.json") -> list[PatientDocument]:
    """Load documents from JSON file."""
    filepath = DOCUMENTS_DIR / filename
    if not filepath.exists():
        logger.warning(f"Document file not found: {filepath}. Using sample documents.")
        return SAMPLE_DOCUMENTS

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    documents = [PatientDocument.from_dict(d) for d in data]
    logger.info(f"Loaded {len(documents)} documents from {filepath}")
    return documents


def get_documents_by_category(documents: list[PatientDocument], category: str) -> list[PatientDocument]:
    """Filter documents by category."""
    return [doc for doc in documents if doc.category == category]


def get_document_stats(documents: list[PatientDocument]) -> dict:
    """Get statistics about the document collection."""
    stats = {
        "total_documents": len(documents),
        "total_words": sum(doc.word_count for doc in documents),
        "by_category": {},
        "categories": list(set(doc.category for doc in documents)),
    }

    for category in stats["categories"]:
        cat_docs = [d for d in documents if d.category == category]
        stats["by_category"][category] = {
            "count": len(cat_docs),
            "total_words": sum(d.word_count for d in cat_docs),
            "topics": list(set(d.topic for d in cat_docs)),
        }

    return stats


def add_custom_document(
    doc_id: str,
    title: str,
    category: str,
    topic: str,
    source: str,
    english_text: str,
    source_url: str = None,
) -> PatientDocument:
    """Create a new custom document."""
    word_count = len(english_text.split())
    return PatientDocument(
        doc_id=doc_id,
        title=title,
        category=category,
        topic=topic,
        source=source,
        source_url=source_url,
        english_text=english_text,
        word_count=word_count,
    )


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_documents():
    """Initialize the document collection - save samples if no documents exist."""
    json_path = DOCUMENTS_DIR / "source_documents.json"

    if not json_path.exists():
        logger.info("No existing documents found. Initializing with sample documents.")
        save_documents_to_json(SAMPLE_DOCUMENTS)
        return SAMPLE_DOCUMENTS
    else:
        return load_documents_from_json()


if __name__ == "__main__":
    # Initialize and display stats
    documents = initialize_documents()
    stats = get_document_stats(documents)

    print("\n" + "="*60)
    print("DOCUMENT COLLECTION STATISTICS")
    print("="*60)
    print(f"\nTotal Documents: {stats['total_documents']}")
    print(f"Total Word Count: {stats['total_words']:,}")
    print(f"\nCategories: {len(stats['categories'])}")

    for category, cat_stats in stats['by_category'].items():
        print(f"\n  {category.upper()}:")
        print(f"    Documents: {cat_stats['count']}")
        print(f"    Words: {cat_stats['total_words']:,}")
        print(f"    Topics: {', '.join(cat_stats['topics'])}")

    print("\n" + "="*60)
    print("DOCUMENT LIST")
    print("="*60)
    for doc in documents:
        print(f"\n[{doc.doc_id}] {doc.title}")
        print(f"    Category: {doc.category} | Topic: {doc.topic}")
        print(f"    Source: {doc.source} | Words: {doc.word_count}")
