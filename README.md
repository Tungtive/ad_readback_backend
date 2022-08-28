# **AD-RedBack**
## This is the repository for COMP90082-2022-SM2 group AD-RedBack
## To clone this repo, please use the link below:
https://github.com/COMP90082-2022-SM2/AD-RedBack.git



## About the project
<br>

### **Project Description**
Human movement disorders affect one-third of Australians. However, the traditional methods of measuring movement, such as video motion capture systems, are expensive and have many limitations. Although we already have a deep neural networks model that can convert the data of wearable internal measurement units (IMUs) into ankle joint angles, which is low-cost and remove the restriction, this model still cannot be directly available to non-IT users. This project will build a user-friendly system for this model so that it can be applied to more daily and research scenarios.

<br>

### **Client**
**A/Prof David Ackland**

A/Prof Ackland's background is in biomechanics and orthopaedics. His research focuses on computational modelling and experimentation of human movement biomechanics, with a particular emphasis on the structure and function of the upper and lower extremities, and jaw. He employs medical imaging, human motion experiments, musculoskeletal modelling, and in vitro biomechanical experiments as his primary research techniques. His research outputs to date have had a strong focus on the measurement and modelling of muscle and joint function during human movement, as well as the design and evaluation of joint replacement prostheses.

Source: [David Ackland](https://findanexpert.unimelb.edu.au/profile/57084-david-ackland)<br>

**Damith Senanayake**

Damith Asanka Senanayake is a research fellow In artificial intelligence and biomedical engineering. He is the first author of the paper Real-time conversion of inertial measurement unit data to ankle joint angles using deep neural networks. The model in this paper is also the core conversion model to be used in this project.

Source: [Damith Asanka Senanayake](https://findanexpert.unimelb.edu.au/profile/850537-damith-asanka-senanayake)<br>
<br>

### **Development Environment**
#### **Developing Languages**
**Back-end** developing language: Python<br>
**Front-end** developing language: HTML5
#### **Management Tools**
**Confluence**: [link](https://confluence.cis.unimelb.edu.au:8443/display/COMP900822022SM2ADRedBack/COMP90082-2022-SM2-AD-RedBack+Home)<br>
**Trello**:[link](https://trello.com/invite/b/6T6SYPZp/5427bf694ab92738651b741df92883cb/kanban)

#### **Environment Setup**
Since we use docker container to run our code, so you need install Docker before run the project.<br><br>
After you clone the repo and install the Docker, just run the code below at the root of the project.<br>
```bash
docker-compose up # launches everything (database + flask)
```
Check if Flask is running by reaching localhost:5000 in your browser. And to check react, just by reaching localhost:3000 in your browser.<br>

You can make some changes in the flask folder and after each save, the app will automatically detect the changes and reload (sometimes you'll need to refresh the webpage ...).

To stop everything :
```bash
docker-compose down # or crtl c ...
```

<br>

### **Open Source Agreement**
This project is not an open source project. **No one** can disclose the source code of the project to personnel unrelated to the project. All team members should abide by the [academic integrity policy of the University of Melbourne](https://academicintegrity.unimelb.edu.au/)

<br>

### **Project Schedule**
**2022.8.20**<br>
At present, we have obtained the source code of the neural network model provided by the customer, communicated the project requirements with the customer, and completed the requirements analysis and conceptual design. Software development will begin in the next sprint.

**2022.8.27**<br>
React+flask+mongodb+docker framework was created.