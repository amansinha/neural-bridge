FROM braintok/self-attention-agent:CarRacing
RUN apt install tmux -y
WORKDIR /opt/app
ADD . /opt/app
RUN pip3 install pyzmq
RUN rm /usr/local/lib/python3.5/dist-packages/gym/envs/box2d/car_racing.py
RUN rm /usr/local/lib/python3.5/dist-packages/gym/envs/box2d/car_dynamics.py
RUN pip3 install gym==0.9.4
RUN pip3 install tensorflow==1.13.1
RUN pip3 install scipy==1.2
COPY ./car_racing.py /usr/local/lib/python3.5/dist-packages/gym/envs/box2d/
#COPY ./car_dynamics.py /usr/local/lib/python3.5/dist-packages/gym/envs/box2d