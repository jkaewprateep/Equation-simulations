# Equation-simulations
How do you create equation simulation ? We see from the Flappy birds game that we need to transfrom its input as { K_w and K_h } for { press_W and press_hold } to create a movement of the Flappy bird player in the Flappy bird games. How we create the equation that the AI deep learning models understand and perform the actions until it can play roles as auto-pilot auto-play ?

👧💬 We create equations easy as this if you talking to them the scientist they transforms your requirements into mathametics equations.
1. The stage has limits of the ceiling and floor.
2. The satge has the limits of the top and buttom pipe gap.
3. Flappy bird play hits one of the upper or lower gap is to new continue.
4. Passing though each gap at the specific point the games give a reward for the Flappy birds.

## Approches ##

We create equation to AI machine learning to learn our conditions and the AI try to play follow our rules with the simple models, if they play well that is because your inputs is the correct conditions but if not we can discuss your approch and simulation.

#### Player avoid buttom gap ####

👧💬 Too close return the mimus and small value, you can control only one variable which is 'player_y_array' what you todo when you are the AI !?
```
( next_pipe_bottom_y_array - player_y_array ) - ( player_y_array - target )
```

#### Player avoid upper gap ####

👧💬 Too close return the minus and small value, you can control only one variable which is 'player_y_array' what you todo when you are the AI !?
```
( player_y_array - next_pipe_top_y_array ) - ( player_y_array - target )
```

#### Player keep velocity balance of continue create distance ####

👧💬 After few rounds play the previous rounds results as the model learning it will try to follow of some rules if the AI can learn of the patterns. The game provide us next_pipe_dist_to_player_array and next_next_pipe_dist_to_player_array which are relative distance, the relative distance is the distance compare to our player in the game you can use this value but to make it easy understanding ``` increase of step is approximates increase of distance ``` . It is optiional to fllied the step into because all together are values and resonse to the game response input but we add step at the backside of the records input to see increasing of its number or you can add the internal step variables if you working with remotes devices. 🧸💬 The reasons is boardcastings medium is public and radios wide ranges is reflecting and travels that is one of the security working with radios must study in fileds area, in room study there are reflection but our electronics devices handlings it. 🐑💬 ``` 75 cm. diameters as examples ``` , 👧💬 ``` The rewards is also optional ```  when working distance is continue . 


```
distance_accum + ( 10 * reward )
```

##### X -axis	#####
 
```           
                   O                   O
               O       O           O       O
           O              O    O              O
---------X---------|---------X---------|---------X---------|
```

##### Y -axis #####

```
XXX------O------XXX|XXX------O------XXX|XXX------O------XXX|
```
	
##### Left side point degrades #####

``` 
leftside_velocity_pt = int( next_pipe_dist_to_player_array  / 2 )
``` 
	
##### right side point degrades #####

```  
rightside_velocity_pt = int( next_pipe_dist_to_player_array  + 
                        ( next_next_pipe_dist_to_player_array - next_pipe_dist_to_player_array ) / 2 ) 
``` 
	
##### position Y upper gap #####

``` 
next_pipe_top_y_array
``` 

##### position Y standard gap #####

``` 
next_pipe_bottom_y_array 
``` 

##### position Y upper gap next #####

``` 
next_next_pipe_top_y_array
``` 

##### position Y standard gap next #####

``` 
next_next_pipe_bottom_y_array
``` 
	
##### distance from player_y_array to upper gap #####

``` 
distance_upper_gap = next_pipe_bottom_y_array - player_y_array - next_pipe_top_y_array
``` 

##### distance from player_y_array to standard gap #####

``` 
distance_standard_gap = next_pipe_bottom_y_array - player_y_array
``` 

##### distance from player_y_array to upper gap next #####

``` 
distance_upper_gap_next = next_next_pipe_bottom_y_array - player_y_array - next_next_pipe_top_y_array
``` 

##### distance from player_y_array to standard gap next #####

``` 
distance_standard_gap_next = next_next_pipe_bottom_y_array - player_y_array
``` 

##### acceleration #####

``` 
accleration = distance_upper_gap * accum_velocity
``` 

##### controls #####

``` 
contrl = distance_accum
contrl_left = step ( optional )
``` 

## Simulation ##

When the sine wave is the Flappy birds games player movement and the rectangle line slope ( straigth line ) is our AI input the press K_w for press W buttom. Our goal is to simulate Flappy bird movement in the scopes of randoms response from the games to prove our equations and idea. The problem is we do not modified the game it remains in the same speed and responses are only from gamestate() and you cannot breaks the game speed to win it with the power of computer speed.

``` 
gameState = {'player_y': 256, 'player_vel': 0, 'next_pipe_dist_to_player': 309.0, 'next_pipe_top_y': 97, 
'next_pipe_bottom_y': 347, 'next_next_pipe_dist_to_player': 453.0, 'next_next_pipe_top_y': 113, 'next_next_pipe_bottom_y': 363} 
```

![Alt text](https://github.com/jkaewprateep/Equation-simulations/blob/main/Figure_1.png?raw=true "Title")

## Simulation - key press ##

Extracting only our key press from the previous graph. It is something they called accelerate press where it is trends at the time at the games luanch they told it make human more fun with game because we are better in this as our neurons does. The purpose is to make AI Deep learning more close to our skills a bit but it is preferred because some tasks the rules conditions cannot do. Some tasks that rules conditions doing well, volumes, meters, matrixes, balance and that rules condition do not performing well for complexing deviding group or catagorized problems compared with in scopes without initail coditions.

| Conditions | AI perform | Human perform | Rules condition |
| --- | --- | --- | --- |
| Catagorize pictures | Better | Best | Conditions |
| Catagorize data | Small improvement | Best, small amount | Conditions |
| Volumes matrixes | Same as rules | Experiences | Best |
| Balances | Small improvement | Experiences | Conditions |
| Acclerelation | Small improvement | Experiences | Conditions |

🐑💬 Do you see similarlity of the both graph of the Flappy birds player and our input press ( K_w ) W buttom ? That explain how do we mapping input and output functions return.
🧸💬 You can try using domain transfroms formula for the method but that is further study if you do it the same as wavelengths response frequency propagation of short Fast furrier transfrom we will study it, speech data and wave is our favourite subject.

![Alt text](https://github.com/jkaewprateep/Equation-simulations/blob/main/Figure_2.png?raw=true "Title")

## Files and Directory ##

1. sample.py : sample graph create.
2. Figure_1.png : graph 1 simulate our expecting movement.
3. Figure_2.png : graph 2 the solve graph.
4. FlappyBird_small.gif : result image.
5. README.md : readme file.

## Result image ##

Result after triaing for periods but sill have problem about longer day running and learning time and etc. we to simulate.

![Alt text](https://github.com/jkaewprateep/Equation-simulations/blob/main/FlappyBird_small.gif?raw=true "Title")

