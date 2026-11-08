Overview plan, with the poker implemntation:

- Each agent gets the same task: Here's your hand and the table, do you fold, call or raise
- Each agent is the same LLM with a different perosnality
- The servo is a guage. It shows what is going inside the agent. If the agent doesn't know what is 
doing then the servo jitters back and forth. If it is settled, the servo points in one direct way
( safe or risky)
- The knob is the steering and it changes the agent's personality while its running without restarting
it. The jittering agent then commits to one direction
- If the LED is green, the agent completes its task or wins the hand in this case
 


We will use activation steering which adds a risk vector (a number that is added from the contrapositive 
prompts and it is fixed). 
The knob only changes the coefficient (c) which scales how much of that vector is added
Negative c means cautios
0 = c means no steering
Positive c means it is reckless
Basically this is the "changing parameters without terminating the agent"



