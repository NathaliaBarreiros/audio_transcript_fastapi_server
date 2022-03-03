# Audio Transcription API based on Mozilla's DeepSpeech library and FastAPI web framework

This service integrates audio transcription feature based on Mozilla's DeepSpeech library. There are two endpoints for uploading audio data, the first one is for uploading .wav audios and the other one is used for uploading audios in base64 encoding format; both endpoints return a CSV file containing either the name of the audios uploaded and their transcription along with a status code 200, if the request was valid. Both endpoints also validate that either the audio file and the Base64 encoding string data uploaded match to a .wav audio file; otherwise, a HTTP exception is thrown with the status code 415 for unsupported media type. Audio uploading endpoints only can be queried from an authorized user; otherwise, a HTTP exception is throw with the status code 401 for unauthorized caller. For the authorization feature, the OAuth2 specification was used; so there is an endpoint for creating a user with its username, email and password, these data is stored in a sqlite database and this table is manage through Tortoise ORM, it's important to mention that hashed password is stored on database. Then, there is other endpoint that generates a JWT token. And only if the token is validated considered the payload and the application secret data, the two audio uploading endpoints described above can be query.

## Try it on FastAPI's interactive API docs!

- POST request for creating a user:

![](/images/step1.png)

![](/images/step1_1.png)

- POST request for generating a JWT:

![](/images/step2.png)

![](/images/step2_1.png)

- Endpoint authorization with OAuth2:

![](/images/step3.png)

![](/images/step3_1.png)

- POST request for uploading .wav audios & getting their transcriptions:

![](/images/step4.png)

![](/images/step4_1.png)

- POST request for uploading .wav audio files' Base64 encoding & getting their transcriptions:

![](/images/step5.png)

![](/images/step5_1.png)
