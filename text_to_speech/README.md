# Robot Vocalizer 

Test Text -> Speech:
```python
python3 src/robot_vocalizer/test.py --deepgram-api-key $DEEPGRAM_API_KEY
```

Test Plan -> Text -> Speech:

```python
python3 src/llm_test.py --openai-api-key $OPENAI_API_KEY --deepgram-api-key $DEEPGRAM_API_KEY
```