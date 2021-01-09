FROM braintok/self-attention-agent:CarRacing
RUN apt install tmux -y
WORKDIR /opt/app
ADD . /opt/app
RUN pip3 install pyzmq
RUN rm /usr/local/lib/python3.5/dist-packages/gym/envs/box2d/car_racing.py
RUN rm /usr/local/lib/python3.5/dist-packages/gym/envs/box2d/car_dynamics.py
COPY ./car_racing.py /usr/local/lib/python3.5/dist-packages/gym/envs/box2d/
COPY ./car_dynamics.py /usr/local/lib/python3.5/dist-packages/gym/envs/box2d

