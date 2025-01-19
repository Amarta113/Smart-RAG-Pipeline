css = '''
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem; 
    margin-bottom: 1rem; 
    display: flex;
    background_color: red;
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTxbnrES2bx8XWdkmuLcWRakmzMr7ino6dVSg&s">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcThm6jtng5blMlCha7BBjNBc4OO788k_ho7lQ&s">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
