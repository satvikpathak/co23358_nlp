const rewriteBtn = document.getElementById('rewrite');
const inputEl = document.getElementById('input');
const outputEl = document.getElementById('output');
const loader = document.getElementById('loader');
const toneSel = document.getElementById('tone');

// Updated to match backend which we run on port 8001
const API_URL = 'http://127.0.0.1:8001/rewrite';

function showLoader(show=true){
  loader.classList.toggle('hidden', !show);
}

rewriteBtn.addEventListener('click', async ()=>{
  const text = inputEl.value.trim();
  const tone = toneSel.value || 'polite';
  if(!text){
    alert('Please enter some text to rewrite.');
    return;
  }
  showLoader(true);
  outputEl.textContent = '';
  try{
    const res = await fetch(API_URL, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({text, tone}),
    });
    if(!res.ok){
      const err = await res.json();
      throw new Error(err.detail || 'Request failed');
    }
    const data = await res.json();
    outputEl.textContent = data.rewritten;
  }catch(err){
    outputEl.textContent = 'Error: ' + err.message;
  }finally{
    showLoader(false);
  }
});
