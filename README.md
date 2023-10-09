# ForensicsChallenge2023 Team6

# Our presantation in [Vienna](https://mmf.univie.ac.at/kharkiv-vienna/vienna-2023/) 

https://docs.google.com/presentation/d/1_GBwiq_9nEkB1opoOX9BCcrERXAkWUT0IQ4eg9GHejQ/edit#slide=id.g285b72f5a0a_0_606
<!---

# Challenge №1
**Use computer vision to isolate each person in a video for 3D recreation or identifying
suspects in crime scenes.**

To solve this task we used several models such as: YOLOv8, Amazon Rekognition, Open-CV. As a result, we saw that the YOLOv8 model is best suited for this task, since it shows the most accurate results and is the fastest at the same time.

Work expample:

**YOLOv8**

https://github.com/zepif/ForensicsChallenge2023_team6/assets/95505468/024d4214-79d8-4b6a-8fad-232b34d30843

**Amazon Rekognition**

https://github.com/zepif/ForensicsChallenge2023_team6/assets/95505468/793d8d04-f133-4c44-b748-8790341d4822

**Open-CV**

https://github.com/zepif/ForensicsChallenge2023_team6/assets/95505468/eb8fa8d6-4231-4811-95f3-72dde50224e1


Also, we made a 3d model of object from Västervik dataset using Meshroom.

This model:



https://github.com/zepif/ForensicsChallenge2023_team6/assets/95505468/8caece61-9fcf-470d-bdfc-40ff7b4795a0




# Challenge №2
**Use computer vision to identify relevant objects or patterns in a photo or video for
criminal investigation.**

For this task we tried to train YOLOv8 model, but due to computer power and AWS SageMaker problems its doesnt show sufficient result. Thus we decided to use pre-trained YOLO model, which can detect only guns.

Work expample:

https://github.com/zepif/ForensicsChallenge2023_team6/assets/95505468/22d0ee8b-1029-4468-96ad-c7a8bd6c6cc7

# Challenge №3
**Use natural language processing to generate a detailed description of a crime scene
for public understanding and potential crime-solving.**

For this task we used ChatGPT. We created a special promt, which allows to generate a description of the crime on the prococol.

Work expample and promt can be found here: chat.html

# Challenge №4
**Use large language models to generate visualizations or text descriptions of a crime
scene or investigation.**

For this task we used ChatGPT and DeepAI. We used ChatGPT to find the most important things in prococol. After, we generate images using DeepAI.

Work expample: 

![generated1](https://github.com/zepif/ForensicsChallenge2023_team6/assets/95505468/59eb2afc-20be-4c99-9d04-30a0abafa9ea)

![generated2](https://github.com/zepif/ForensicsChallenge2023_team6/assets/95505468/473db290-87b4-4163-aad1-cc0b09da28c3)

![generated3](https://github.com/zepif/ForensicsChallenge2023_team6/assets/95505468/58cb9890-3cdc-4641-9383-5c5d1c0beb24)

![generated4](https://github.com/zepif/ForensicsChallenge2023_team6/assets/95505468/8894cea7-479e-4113-a964-0ebaeea8a02c)


# Sound analysis

Also, in addition to the main tasks, we decided to make sound analysis. We think it might help the police if the crime happened in the dead zone of the camera. Our programm can detect shots and and convert the speech from the video camera into text, which can help in generating a description of the crime and in the course of the crime itself.

**Example of speech recognition:**

Video:

https://github.com/zepif/ForensicsChallenge2023_team6/assets/95505468/964c7f0b-741e-4a95-aa2b-d1856ba431f9

(a trimmed version, because github does not allow to download the full version)

Result: transcription.txt

**Example of shots recognition:**

Video:

https://github.com/zepif/ForensicsChallenge2023_team6/assets/95505468/d9e7899b-e7a2-4035-a7ca-1719a3942cf5

Result:
![image](https://github.com/zepif/ForensicsChallenge2023_team6/assets/95505468/38a5fc5d-494c-40af-81b3-3b1c6ac412a1)
also program shows shots time in console
--->
