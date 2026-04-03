from django.shortcuts import render
from django.http import HttpResponse
import time

def index(request):
    return render(request, 'index.html')

def run_algorithm(request, algorithm):
    start = time.perf_counter()
    time.sleep(0.5)
    elapsed = time.perf_counter() - start
    return HttpResponse(f"Алгоритм: {algorithm}. Время выполнения: {elapsed:.4f} сек.")