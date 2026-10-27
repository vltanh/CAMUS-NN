// Branch ordering for reproducible hybrid identifiers.
// See SORT-LICENSE.
#pragma once
#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace camus {

template<class T, class Compare> class ReferenceSort {
    std::vector<T>& data;
    Compare compare;
    bool less(int a,int b) const { return compare(data[a],data[b]); }
    void swap(int a,int b) { std::swap(data[a],data[b]); }
    void insertion(int a,int b) {
        for(int i=a+1;i<b;++i)for(int j=i;j>a&&less(j,j-1);--j)swap(j,j-1);
    }
    void sift(int root,int hi,int first) {
        for(;;) {
            int child=2*root+1;
            if(child>=hi)break;
            if(child+1<hi&&less(first+child,first+child+1))++child;
            if(!less(first+root,first+child))return;
            swap(first+root,first+child);root=child;
        }
    }
    void heap(int a,int b) {
        int hi=b-a;
        for(int i=(hi-1)/2;i>=0;--i)sift(i,hi,a);
        for(int i=hi-1;i>=0;--i){swap(a,a+i);sift(0,i,a);}
    }
    void reverse(int a,int b) { for(int i=a,j=b-1;i<j;++i,--j)swap(i,j); }
    std::pair<int,bool> partition(int a,int b,int pivot) {
        swap(a,pivot);int i=a+1,j=b-1;
        while(i<=j&&less(i,a))++i;
        while(i<=j&&!less(j,a))--j;
        if(i>j){swap(j,a);return {j,true};}
        swap(i,j);++i;--j;
        for(;;) {
            while(i<=j&&less(i,a))++i;
            while(i<=j&&!less(j,a))--j;
            if(i>j)break;
            swap(i,j);++i;--j;
        }
        swap(j,a);return {j,false};
    }
    int partition_equal(int a,int b,int pivot) {
        swap(a,pivot);int i=a+1,j=b-1;
        for(;;) {
            while(i<=j&&!less(a,i))++i;
            while(i<=j&&less(a,j))--j;
            if(i>j)break;
            swap(i,j);++i;--j;
        }
        return i;
    }
    bool partial_insertion(int a,int b) {
        int i=a+1;
        for(int step=0;step<5;++step) {
            while(i<b&&!less(i,i-1))++i;
            if(i==b)return true;
            if(b-a<50)return false;
            swap(i,i-1);
            if(i-a>=2)for(int j=i-1;j>=1;--j){if(!less(j,j-1))break;swap(j,j-1);}
            if(b-i>=2)for(int j=i+1;j<b;++j){if(!less(j,j-1))break;swap(j,j-1);}
        }
        return false;
    }
    void break_patterns(int a,int b) {
        int length=b-a;
        if(length<8)return;
        uint64_t random=length,modulus=1;
        while(modulus<=uint64_t(length))modulus<<=1;
        for(int i=a+(length/4)*2-1;i<=a+(length/4)*2+1;++i) {
            random^=random<<13;random^=random>>7;random^=random<<17;
            int other=int(random&(modulus-1));
            if(other>=length)other-=length;
            swap(i,a+other);
        }
    }
    void order2(int& a,int& b,int& swaps) const { if(less(b,a)){std::swap(a,b);++swaps;} }
    int median(int a,int b,int c,int& swaps) const {
        order2(a,b,swaps);order2(b,c,swaps);order2(a,b,swaps);return b;
    }
    std::pair<int,int> pivot(int a,int b) const {
        int length=b-a,swaps=0,i=a+length/4,j=a+(length/4)*2,k=a+(length/4)*3;
        if(length>=8) {
            if(length>=50){i=median(i-1,i,i+1,swaps);j=median(j-1,j,j+1,swaps);k=median(k-1,k,k+1,swaps);}
            j=median(i,j,k,swaps);
        }
        return {j,swaps==0?1:swaps==12?2:0};
    }
    void pdq(int a,int b,int limit) {
        bool balanced=true,partitioned=true;
        for(;;) {
            int length=b-a;
            if(length<=12){insertion(a,b);return;}
            if(limit==0){heap(a,b);return;}
            if(!balanced){break_patterns(a,b);--limit;}
            auto [p,hint]=pivot(a,b);
            if(hint==2){reverse(a,b);p=(b-1)-(p-a);hint=1;}
            if(balanced&&partitioned&&hint==1&&partial_insertion(a,b))return;
            if(a>0&&!less(a-1,p)){a=partition_equal(a,b,p);continue;}
            auto [mid,already]=partition(a,b,p);partitioned=already;
            int left=mid-a,right=b-mid,threshold=length/8;
            if(left<right){balanced=left>=threshold;pdq(a,mid,limit);a=mid+1;}
            else{balanced=right>=threshold;pdq(mid+1,b,limit);b=mid;}
        }
    }
public:
    ReferenceSort(std::vector<T>& values,Compare comparator):data(values),compare(comparator){}
    void run() {
        int limit=0;
        for(size_t n=data.size();n;n>>=1)++limit;
        pdq(0,int(data.size()),limit);
    }
};

template<class T,class Compare> void reference_sort(std::vector<T>& data,Compare compare) {
    ReferenceSort<T,Compare>(data,compare).run();
}

} // namespace camus
