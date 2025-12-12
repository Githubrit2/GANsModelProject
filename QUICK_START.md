# Quick Start Guide

## ✅ Step 1: Test Model Loading

```bash
cd webapp
python test_model.py
```

You should see:
```
[OK] Model import successful
[OK] Generator created
[OK] Model loaded successfully
[SUCCESS] Everything works!
```

## ✅ Step 2: Run Flask App

```bash
python flask_app.py
```

You should see:
```
Starting Flask App...
[OK] Model loaded successfully!
* Running on http://127.0.0.1:5000
```

## ✅ Step 3: Open Browser

Go to: **http://localhost:5000**

## ✅ Step 4: Generate Face

1. Type: "young woman"
2. Click "Generate Face"
3. Wait 2-3 seconds
4. See your generated face!

## 🔧 Troubleshooting

### If you see "Failed to fetch":

1. **Check Flask is running** - Look at terminal, should say "Running on..."
2. **Check browser console** - Press F12, look for errors
3. **Try different browser** - Chrome/Firefox/Edge
4. **Restart Flask** - Press Ctrl+C, run again

### If model doesn't load:

```bash
# Check model exists
dir ..\outputs\generator_epoch_100.pth
```

### Still not working?

1. Stop Flask (Ctrl+C)
2. Run test: `python test_model.py`
3. Share the output

## 📝 What to Check in Browser Console (F12)

Look for:
- "Sending request with text: ..."
- "Response status: 200"
- "Received data: ..."
- "Image displayed successfully"

If you see errors, share them!
