from django.shortcuts import render
from django.conf import settings
from .forms import UploadImageForm
from .utils.predictor import predict_and_save

def upload_view(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            img_bytes = img_file.read()
            rel = predict_and_save(img_bytes)   # e.g. 'results/result_abc.jpg'
            result_url = settings.MEDIA_URL + rel
            return render(request, 'mask/result.html', {'result_url': result_url})
    else:
        form = UploadImageForm()
    return render(request, 'mask/upload.html', {'form': form})
