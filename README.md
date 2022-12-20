# Equation-simulations
How do you create equation simulation ? We see from the Flappy birds game that we need to transfrom input as { K_w and K_h } for { press_W and press_hold } to create a movement of the Flappy bird player in the Flappy bird games. How we create the equation that the AI deep learning models understand and perform the actions until it can play roels as auto-pilo auto-play ?

👧💬 We create equations easy as this if you talking to them the scientist they transforms your requirements into matchametics equations.
1. The stage has limits of the ceilning and floor.
2. The satge has the limits of the top and buttom pipe gap.
3. Flappy bird play hits one of the upper of lower gap is to new continue.
4. Passing thouse each gap at the specific point the games give a reward for the Flappy birds.

## Approches ##

We create equation to AI machine learning to learn our conditions and the AI try to play follow our rules with the simple models, if they play well that is because your inputs is the correct conditions but if not we can discuss your approch and simulation 

#### Player avoid buttom gap ####

👧💬 Too close return the mimus and small value, you can control only one variable which is 'player_y_array' what you todo when you are the AI !?
```
( next_pipe_bottom_y_array - player_y_array ) - ( player_y_array - target )
```

#### Player avoid upper gap ####

👧💬 Too close return the mimus and small value, you can control only one variable which is 'player_y_array' what you todo when you are the AI !?
```
( player_y_array - next_pipe_top_y_array ) - ( player_y_array - target )
```

#### Player keep velocity balance of continue create distance ####

👧💬 After few rounds play the previous rounds results as the model learning it will try to follow of some rules if the AI can learn of the patterns.
```
distance_accum + ( 10 * reward )
```

![Alt text](https://github.com/jkaewprateep/Equation-simulations/blob/main/Figure_1.png?raw=true "Title")

## Files and Directory ##

1. sample.py : sample graph create
2. Figure_1.png : graph 1 simulate our expecting movement 
3. Figure_2.png : graph 2 the solve graph
4. FlappyBird_small.gif : result image
5. README.md : readme file

## Result image ##

![Alt text](https://github.com/jkaewprateep/Equation-simulations/blob/main/Figure_2.png?raw=true "Title")

![Alt text](https://github.com/jkaewprateep/Equation-simulations/blob/main/FlappyBird_small.gif?raw=true "Title")

